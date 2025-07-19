import os
import sys
import importlib

# Set working directory to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Optional: Set path to ensure local modules are discoverable
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def import_and_run(module_name: str, phase_label: str):
    print(f"\n🟦 Starting {phase_label} from {module_name}.py")
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "run"):
            mod.run()
            print(f"✅ Completed {phase_label}")
        else:
            print(f"❌ {module_name}.py is missing a 'run()' function.")
    except ModuleNotFoundError:
        print(f"❌ Could not find file: {module_name}.py")
    except Exception as e:
        print(f"❌ Error in {module_name}.py: {e}")

def main():
    print("📦 HS Code Enrichment Pipeline – Full Workflow")

    import_and_run("phase1_prepare_prompts", "Phase 1: Prompt Generation")
    import_and_run("phase2_llm_inference", "Phase 2: LLM Inference")
    import_and_run("phase3_postprocessing", "Phase 3: Post-Processing")

    print("\n💡 To launch Phase 4 (Streamlit UI), run this command manually:")
    print(f"    streamlit run {os.path.join(BASE_DIR, 'phase4_streamlit_review.py')}\n")

if __name__ == "__main__":
    main()
