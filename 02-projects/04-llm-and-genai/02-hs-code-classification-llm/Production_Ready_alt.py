import os
import re
import time
import random
import subprocess
import pandas as pd
import streamlit as st
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from llama_cpp import Llama

# === Configuration ===
@dataclass
class Config:
    BASE_DIR: str = "C:/Data/Job/UK/Zencargo"
    EXCEL_FILE: str = os.path.join(BASE_DIR, "Automating_HS_Code_validation.xlsx")
    HMRC_FILE: str = os.path.join(BASE_DIR, "uk-tariff-2021-01-01--v4.0.1060--commodities-report.ods")
    VALIDATED_CSV: str = os.path.join(BASE_DIR, "validated_output.csv")
    PROMPT_CSV: str = os.path.join(BASE_DIR, "prompts_for_llm.csv")
    ENRICHED_CSV: str = os.path.join(BASE_DIR, "llm_enriched_output.csv")
    LLAMA_CLI: str = os.path.join(BASE_DIR, "LLAMA-cpu-x64", "llama-cli.exe")
    MODEL_PATH: str = os.path.join(BASE_DIR, "LLAMA-cpu-x64", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    
    # Production settings
    LOG_FILE: str = os.path.join(BASE_DIR, "processing.log")
    ERROR_LOG: str = os.path.join(BASE_DIR, "errors.json")
    METRICS_FILE: str = os.path.join(BASE_DIR, "metrics.json")
    BACKUP_DIR: str = os.path.join(BASE_DIR, "backups")
    
    # Processing limits
    MAX_ROWS_PER_BATCH: int = 100
    LLM_TIMEOUT: int = 120
    MAX_RETRIES: int = 3
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5

# === Logging Setup ===
def setup_logging(config: Config):
    """Setup comprehensive logging"""
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# === Data Validation ===
class DataValidator:
    """Comprehensive data validation and quality checks"""
    
    @staticmethod
    def validate_hs_code_format(code: str) -> bool:
        """Validate HS code format (10 digits)"""
        if not code or pd.isna(code):
            return False
        cleaned = re.sub(r"[^\d]", "", str(code))
        return len(cleaned) == 10 and cleaned.isdigit()
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
        """Validate required columns exist"""
        missing_cols = [col for col in required_cols if col not in df.columns]
        return missing_cols
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data quality check"""
        quality_report = {
            "total_rows": int(len(df)),
            "missing_sku": int(df["sku_code"].isna().sum()),
            "missing_hs_code": int(df["hs_code"].isna().sum()),
            "invalid_hs_format": 0,
            "duplicate_skus": int(df["sku_code"].duplicated().sum()),
            "empty_strings": 0,
            "quality_score": 0.0
        }
        
        # Check HS code format
        for code in df["hs_code"]:
            if not DataValidator.validate_hs_code_format(code):
                quality_report["invalid_hs_format"] += 1
        
        # Check empty strings
        quality_report["empty_strings"] = (
            (df["sku_code"] == "").sum() + 
            (df["hs_code"] == "").sum()
        )
        
        # Calculate quality score
        total_issues = (
            quality_report["missing_sku"] + 
            quality_report["missing_hs_code"] + 
            quality_report["invalid_hs_format"] + 
            quality_report["duplicate_skus"] + 
            quality_report["empty_strings"]
        )
        quality_report["quality_score"] = max(0, 1 - (total_issues / quality_report["total_rows"]))
        
        return quality_report
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=["sku_code"])
        
        # Standardize text fields
        df["sku_code"] = df["sku_code"].str.upper().str.strip()
        if "product_description" in df.columns:
            df["product_description"] = df["product_description"].str.strip()
        
        # Handle missing values
        df["hs_code"] = df["hs_code"].fillna("")
        
        return df

# === Error Handling ===
class ErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.error_log = []
    
    def log_error(self, error: Exception, context: str, sku_code: str = None):
        """Log error with context"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "sku_code": sku_code
        }
        self.error_log.append(error_entry)
        self.logger.error(f"Error in {context}: {error}")
        
        # Save error log
        os.makedirs(os.path.dirname(self.config.ERROR_LOG), exist_ok=True)
        with open(self.config.ERROR_LOG, 'w') as f:
            json.dump(self.error_log, f, indent=2)
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if operation should be retried"""
        if retry_count >= self.config.MAX_RETRIES:
            return False
        
        # Retry on specific errors
        retryable_errors = (TimeoutError, ConnectionError, subprocess.TimeoutExpired)
        return isinstance(error, retryable_errors)

# === Metrics & Monitoring ===
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

class MetricsCollector:
    """Collect and track performance metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = {
            "processing_start": datetime.now().isoformat(),
            "phases": {},
            "performance": {},
            "quality": {},
            "errors": []
        }
    
    def record_phase_metrics(self, phase: str, metrics: Dict):
        """Record metrics for a processing phase"""
        # Convert numpy types to native Python types for JSON serialization
        converted_metrics = convert_numpy_types(metrics)
        
        self.metrics["phases"][phase] = {
            "timestamp": datetime.now().isoformat(),
            **converted_metrics
        }
    
    def record_performance(self, operation: str, duration: float):
        """Record performance metrics"""
        if operation not in self.metrics["performance"]:
            self.metrics["performance"][operation] = []
        self.metrics["performance"][operation].append({
            "timestamp": datetime.now().isoformat(),
            "duration": duration
        })
    
    def record_error(self, error: Exception, context: str):
        """Record error metrics"""
        self.metrics["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "context": context
        })
    
    def save_metrics(self):
        """Save metrics to file"""
        self.metrics["processing_end"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.config.METRICS_FILE), exist_ok=True)
        with open(self.config.METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# === Enhanced HS Code Validation ===
class HSCodeValidator:
    """Enhanced HS code validation with caching"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.valid_codes_cache = None
        self.last_cache_update = None
    
    @lru_cache(maxsize=1)
    def load_hmrc_data(self, hmrc_file: str) -> set:
        """Load and cache HMRC tariff data"""
        try:
            self.logger.info("Loading HMRC tariff data...")
            hmrc_df = pd.read_excel(hmrc_file, engine="odf")
            hmrc_df["reference_hs_code"] = hmrc_df["commodity__code"].astype(str).str.zfill(10)
            valid_codes = set(hmrc_df["reference_hs_code"])
            
            self.logger.info(f"Loaded {len(valid_codes)} valid HS codes")
            return valid_codes
            
        except Exception as e:
            self.logger.error(f"Failed to load HMRC data: {e}")
            raise
    
    def validate_code(self, code: str, valid_codes: set) -> Dict:
        """Enhanced HS code validation with detailed results"""
        if pd.isna(code) or not code:
            return {
                "status": "missing",
                "confidence": 0.0,
                "message": "No HS code provided"
            }
        
        code_str = str(code)
        cleaned = re.sub(r"[^\d]", "", code_str)
        
        if len(cleaned) != 10:
            return {
                "status": "invalid_format",
                "confidence": 0.0,
                "message": f"Invalid format: expected 10 digits, got {len(cleaned)}"
            }
        
        if cleaned not in valid_codes:
            return {
                "status": "not_found",
                "confidence": 0.3,
                "message": "Code not found in HMRC tariff"
            }
        
        return {
            "status": "valid",
            "confidence": 1.0,
            "message": "Valid HS code"
        }

# === Enhanced LLM Integration ===
class LLMProcessor:
    """Enhanced LLM processing with better error handling"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.llm = None
        self.example_pool = self._load_example_pool()
    
    def _load_example_pool(self) -> List[Tuple[str, str]]:
        """Load comprehensive example pool"""
        return [
            ("Men's cotton shirt (Material: Cotton; Use: Clothing; Dimensions: Size M)", "6205200000"),
            ("Bluetooth speaker (Material: Plastic and metal; Use: Electronics; Dimensions: 10x5x5 cm)", "8518210000"),
            ("Toy car (Material: Plastic; Use: Children's toy; Dimensions: 15x7x6 cm)", "9503007000"),
            ("Aluminum laptop stand (Material: Aluminum; Use: Office equipment; Dimensions: 30x25x5 cm)", "7616999099"),
            ("Ceramic mug (Material: Ceramic; Use: Drinkware; Volume: 300ml)", "6912002310"),
            ("Winter gloves (Material: Wool; Use: Clothing; Size: L)", "6116930000"),
            ("Notebook computer (Material: Plastic and metal; Use: Electronics; Screen: 13-inch)", "8471300000"),
            ("Wooden chair (Material: Wood; Use: Furniture; Dimensions: 80x45x45 cm)", "9401610000"),
            ("Leather wallet (Material: Leather; Use: Personal accessory; Dimensions: 10x8x1 cm)", "4202310000"),
            ("LED desk lamp (Material: Plastic and glass; Use: Lighting; Height: 40 cm)", "9405209990"),
            ("Steel hammer (Material: Steel with rubber grip; Use: Hand tool; Length: 30 cm)", "8205200000"),
            ("Men's leather shoes (Material: Leather; Use: Footwear; Size: 42 EU)", "6403511100"),
            ("Plastic food container (Material: Polypropylene; Use: Kitchenware; Capacity: 1.5L)", "3924100000"),
            ("Smartphone (Material: Glass and metal; Use: Communication; Screen: 6.1 inch)", "8517120000"),
            ("Cotton t-shirt (Material: 100% cotton; Use: Casual wear; Size: L)", "6109100000"),
            ("Coffee maker (Material: Stainless steel and plastic; Use: Kitchen appliance; Capacity: 1.2L)", "8516710000"),
            ("Running shoes (Material: Synthetic and rubber; Use: Athletic footwear; Size: 42)", "6403190000"),
            ("Backpack (Material: Nylon; Use: Travel accessory; Capacity: 30L)", "4202120000"),
            ("USB cable (Material: Copper and plastic; Use: Electronic accessory; Length: 1m)", "8544490000"),
            ("Glass vase (Material: Glass; Use: Decorative; Height: 25cm)", "7013370000")
        ]
    
    def initialize_llm(self, model_path: str):
        """Initialize LLM with proper error handling"""
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=8,
                chat_format="llama-2",
                verbose=False  # Reduce verbose output
            )
            self.logger.info("LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def create_enhanced_prompt(self, row: pd.Series) -> str:
        """Create enhanced prompt with better context"""
        sku = row.get("sku_code", "Unknown SKU")
        
        # Handle NaN values and convert to string
        if pd.isna(sku):
            sku = "Unknown_SKU"
        else:
            sku = str(sku)
        
        # Generate deterministic metadata based on SKU
        sku_hash = hashlib.md5(sku.encode()).hexdigest()
        hash_int = int(sku_hash[:8], 16)
        
        # Use hash to select category and product
        categories = ["electronics", "clothing", "furniture", "tools", "kitchenware"]
        category = categories[hash_int % len(categories)]
        
        products = {
            "electronics": ["phone", "laptop", "speaker", "camera"],
            "clothing": ["shirt", "pants", "shoes", "hat"],
            "furniture": ["chair", "table", "desk", "bed"],
            "tools": ["hammer", "screwdriver", "wrench", "drill"],
            "kitchenware": ["mug", "plate", "bowl", "spoon"]
        }
        
        product = products[category][hash_int % len(products[category])]
        
        full_description = f"{sku} - {product} (Material: Various; Use: {category}; Dimensions: Standard)"
        
        instruction = (
            "You are a customs classification specialist for UK import/export operations.\n"
            "Analyze the product description and assign the most precise 10-digit UK HS code.\n"
            "Consider: material composition, primary function, and physical characteristics.\n"
            "Respond with only the 10-digit HS code - no additional text or explanations.\n\n"
            "### Product Description:\n"
            f"{full_description}\n\n"
            "### HS Code:"
        )
        
        return f"[INST] {instruction.strip()} [/INST]"
    
    def process_with_llm(self, prompt: str, retry_count: int = 0) -> Dict:
        """Process prompt with LLM and handle errors"""
        try:
            start_time = time.time()
            
            output = self.llm(
                prompt,
                max_tokens=128,
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.9,
                repeat_penalty=1.2,
                stop=["\n", "</s>"]
            )["choices"][0]["text"].strip()
            
            duration = time.time() - start_time
            
            # Extract HS code
            hs_code = self._extract_hs_code(output)
            
            return {
                "success": True,
                "output": output,
                "hs_code": hs_code,
                "duration": duration,
                "confidence": 0.8 if hs_code else 0.1
            }
            
        except Exception as e:
            self.logger.error(f"LLM processing error: {e}")
            if retry_count < self.config.MAX_RETRIES:
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.process_with_llm(prompt, retry_count + 1)
            
            return {
                "success": False,
                "output": f"Error: {str(e)}",
                "hs_code": None,
                "duration": 0,
                "confidence": 0.0
            }
    
    def _extract_hs_code(self, text: str) -> Optional[str]:
        """Enhanced HS code extraction"""
        # Try exact 10-digit number
        match = re.search(r"\b\d{10}\b", text)
        if match:
            return match.group(0)
        
        # Try within HS Code label
        match = re.search(r"HS\s*Code\s*[:\-]?\s*(\d{10})", text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try other common patterns
        patterns = [
            r"Code[:\s]*(\d{10})",
            r"HS[:\s]*(\d{10})",
            r"Tariff[:\s]*(\d{10})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

# === Main Processing Pipeline ===
class HSCodeProcessor:
    """Main processing pipeline with all improvements"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.error_handler = ErrorHandler(config, self.logger)
        self.metrics = MetricsCollector(config)
        self.validator = DataValidator()
        self.hs_validator = HSCodeValidator(config, self.logger)
        self.llm_processor = LLMProcessor(config, self.logger)
    
    def run_phase1_load_and_normalize(self, product_file: str) -> pd.DataFrame:
        """Enhanced Phase 1 with comprehensive validation"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Phase 1: Load and Normalize")
            
            # Load data
            df = pd.read_excel(product_file)
            
            # Validate required columns
            required_cols = ["sku_code", "hs_code"]
            missing_cols = self.validator.validate_required_columns(df, required_cols)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            df = self.validator.clean_data(df)
            
            # Data quality check
            quality_report = self.validator.check_data_quality(df)
            self.logger.info(f"Data quality report: {quality_report}")
            
            # Normalize HS codes
            def normalize_hs_code(code):
                if pd.isna(code):
                    return None
                cleaned = re.sub(r"[^\d]", "", str(code))
                return cleaned.zfill(10) if len(cleaned) <= 10 else cleaned[:10]
            
            df["normalized_hs_code"] = df["hs_code"].apply(normalize_hs_code)
            df["hs_code_status"] = df["normalized_hs_code"].apply(
                lambda x: "missing" if x is None else "partial" if len(str(x)) < 10 else "complete"
            )
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_performance("phase1_load_normalize", duration)
            self.metrics.record_phase_metrics("phase1", {
                "input_rows": len(df),
                "quality_report": quality_report,
                "duration": duration
            })
            
            self.logger.info(f"Phase 1 completed in {duration:.2f} seconds")
            return df
            
        except Exception as e:
            self.error_handler.log_error(e, "Phase 1", None)
            self.metrics.record_error(e, "Phase 1")
            raise
    
    def run_phase2_validate(self, df: pd.DataFrame, hmrc_file: str) -> pd.DataFrame:
        """Enhanced Phase 2 with caching and detailed validation"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Phase 2: Validate Against HMRC Data")
            
            # Load HMRC data with caching
            valid_codes = self.hs_validator.load_hmrc_data(hmrc_file)
            
            # Validate each code
            validation_results = []
            for idx, row in df.iterrows():
                result = self.hs_validator.validate_code(
                    row["normalized_hs_code"], 
                    valid_codes
                )
                validation_results.append(result)
            
            # Add validation results to dataframe
            df["validation_result"] = [r["status"] for r in validation_results]
            df["validation_confidence"] = [r["confidence"] for r in validation_results]
            df["validation_message"] = [r["message"] for r in validation_results]
            df["needs_enrichment"] = df["validation_result"].isin(["missing", "invalid_format", "not_found"])
            
            # Save validated results
            df.to_csv(self.config.VALIDATED_CSV, index=False)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_performance("phase2_validate", duration)
            self.metrics.record_phase_metrics("phase2", {
                "input_rows": len(df),
                "valid_codes": (df["validation_result"] == "valid").sum(),
                "needs_enrichment": df["needs_enrichment"].sum(),
                "duration": duration
            })
            
            self.logger.info(f"Phase 2 completed in {duration:.2f} seconds")
            return df
            
        except Exception as e:
            self.error_handler.log_error(e, "Phase 2", None)
            self.metrics.record_error(e, "Phase 2")
            raise
    
    def run_phase3_enrich(self, df: pd.DataFrame, max_rows: int = None) -> pd.DataFrame:
        """Enhanced Phase 3 with better LLM integration"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Phase 3: Enrich with LLM")
            
            # Filter rows needing enrichment
            enrichment_rows = df[df["needs_enrichment"]].copy()
            if max_rows:
                enrichment_rows = enrichment_rows.head(max_rows)
            
            if enrichment_rows.empty:
                self.logger.warning("No rows need enrichment")
                return df
            
            # Initialize LLM
            self.llm_processor.initialize_llm(self.config.MODEL_PATH)
            
            # Process each row
            results = []
            total_rows = len(enrichment_rows)
            self.logger.info(f"Starting LLM enrichment for {total_rows} rows")
            
            for idx, row in enrichment_rows.iterrows():
                sku = row.get('sku_code', 'Unknown')
                if pd.isna(sku):
                    sku = "Unknown_SKU"
                self.logger.info(f"Processing SKU: {sku} ({idx + 1}/{total_rows})")
                
                # Create enhanced prompt
                prompt = self.llm_processor.create_enhanced_prompt(row)
                
                # Process with LLM
                result = self.llm_processor.process_with_llm(prompt)
                
                # Update row with results
                enrichment_rows.at[idx, "llm_response"] = result["output"]
                enrichment_rows.at[idx, "parsed_hs_code"] = result["hs_code"] if result["hs_code"] else "❌ Invalid"
                enrichment_rows.at[idx, "llm_confidence"] = result["confidence"]
                enrichment_rows.at[idx, "processing_duration"] = result["duration"]
                
                # Log result
                if result["hs_code"]:
                    self.logger.info(f"✅ Generated HS Code: {result['hs_code']} (confidence: {result['confidence']:.2f})")
                else:
                    self.logger.warning(f"❌ Failed to generate HS Code for {sku}")
                
                results.append(result)
            
            # Update original dataframe
            df.update(enrichment_rows)
            
            # Save enriched results
            df.to_csv(self.config.ENRICHED_CSV, index=False)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_performance("phase3_enrich", duration)
            self.metrics.record_phase_metrics("phase3", {
                "input_rows": len(enrichment_rows),
                "successful_enrichments": sum(1 for r in results if r["success"]),
                "average_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0,
                "duration": duration
            })
            
            self.logger.info(f"Phase 3 completed in {duration:.2f} seconds")
            return df
            
        except Exception as e:
            self.error_handler.log_error(e, "Phase 3", None)
            self.metrics.record_error(e, "Phase 3")
            raise
    
    def run_phase4_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced Phase 4 with sophisticated confidence scoring"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Phase 4: Confidence Scoring")
            
            def calculate_confidence(row):
                """Enhanced confidence calculation"""
                hs_code = str(row.get("parsed_hs_code", "")).strip()
                original = str(row.get("normalized_hs_code", "")).strip()
                llm_confidence = row.get("llm_confidence", 0.0)
                validation_confidence = row.get("validation_confidence", 0.0)
                
                # Base confidence from LLM
                confidence = llm_confidence
                
                # Boost if original and parsed match
                if original and hs_code == original:
                    confidence = min(1.0, confidence + 0.3)
                
                # Boost if first 6 digits match (same chapter)
                elif original and hs_code[:6] == original[:6]:
                    confidence = min(1.0, confidence + 0.2)
                
                # Consider validation confidence
                if validation_confidence > 0:
                    confidence = (confidence + validation_confidence) / 2
                
                # Penalize invalid codes
                if "Invalid" in hs_code or not hs_code.isdigit():
                    confidence *= 0.1
                
                return round(confidence, 3)
            
            # Calculate confidence
            df["confidence"] = df.apply(calculate_confidence, axis=1)
            
            # Determine review needs
            def needs_review(row):
                if row["confidence"] >= self.config.HIGH_CONFIDENCE_THRESHOLD:
                    return False
                if row["confidence"] >= self.config.MEDIUM_CONFIDENCE_THRESHOLD:
                    return "optional"
                return "required"
            
            df["review_status"] = df.apply(needs_review, axis=1)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_performance("phase4_confidence", duration)
            self.metrics.record_phase_metrics("phase4", {
                "total_rows": len(df),
                "high_confidence": (df["confidence"] >= self.config.HIGH_CONFIDENCE_THRESHOLD).sum(),
                "medium_confidence": ((df["confidence"] >= self.config.MEDIUM_CONFIDENCE_THRESHOLD) & 
                                    (df["confidence"] < self.config.HIGH_CONFIDENCE_THRESHOLD)).sum(),
                "low_confidence": (df["confidence"] < self.config.MEDIUM_CONFIDENCE_THRESHOLD).sum(),
                "required_review": (df["review_status"] == "required").sum(),
                "duration": duration
            })
            
            # Save final results
            df.to_csv(self.config.ENRICHED_CSV, index=False)
            
            self.logger.info(f"Phase 4 completed in {duration:.2f} seconds")
            return df
            
        except Exception as e:
            self.error_handler.log_error(e, "Phase 4", None)
            self.metrics.record_error(e, "Phase 4")
            raise
    
    def run_full_pipeline(self, max_rows: int = None) -> pd.DataFrame:
        """Run the complete pipeline with error handling and monitoring"""
        try:
            self.logger.info("Starting full HS Code processing pipeline")
            
            # Phase 1
            df = self.run_phase1_load_and_normalize(self.config.EXCEL_FILE)
            
            # Phase 2
            df = self.run_phase2_validate(df, self.config.HMRC_FILE)
            
            # Phase 3
            df = self.run_phase3_enrich(df, max_rows)
            
            # Phase 4
            df = self.run_phase4_confidence(df)
            
            # Save final metrics
            self.metrics.save_metrics()
            
            self.logger.info("Pipeline completed successfully")
            return df
            
        except Exception as e:
            self.error_handler.log_error(e, "Full Pipeline", None)
            self.metrics.record_error(e, "Full Pipeline")
            self.metrics.save_metrics()  # Save metrics even on failure
            raise

# === Main Execution ===
if __name__ == "__main__":
    config = Config()
    processor = HSCodeProcessor(config)
    
    try:
        # Run with limited rows for testing
        result_df = processor.run_full_pipeline(max_rows=10)
        print(f"✅ Pipeline completed successfully. Processed {len(result_df)} rows.")
        
        # Print summary statistics
        print(f"📊 High confidence predictions: {(result_df['confidence'] >= config.HIGH_CONFIDENCE_THRESHOLD).sum()}")
        print(f"📊 Medium confidence predictions: {((result_df['confidence'] >= config.MEDIUM_CONFIDENCE_THRESHOLD) & (result_df['confidence'] < config.HIGH_CONFIDENCE_THRESHOLD)).sum()}")
        print(f"📊 Low confidence predictions: {(result_df['confidence'] < config.MEDIUM_CONFIDENCE_THRESHOLD).sum()}")
        print(f"📊 Required reviews: {(result_df['review_status'] == 'required').sum()}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}") 