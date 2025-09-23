#!/usr/bin/env python3
"""
Run All Examples
Script untuk menjalankan semua contoh dengan pilihan
"""

import sys
import os
from pathlib import Path
import subprocess
import time

def run_example(script_path, description):
    """Run a single example script"""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(script_path).parent)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False

def show_menu():
    """Show example menu"""
    examples = [
        ("quick_start.py", "Quick Start Test", "Quick system test and basic functionality"),
        ("examples/example_basic_addition.py", "Basic Fish Addition", "Add single fish to database"),
        ("examples/example_batch_processing.py", "Batch Processing", "Process multiple images/species"),
        ("examples/example_database_management.py", "Database Management", "Database utilities and maintenance"),
        ("examples/example_integration_test.py", "Integration Test", "Complete system integration test"),
    ]
    
    print("🐟 Advanced Fish System - Example Runner")
    print("=" * 50)
    print("Available examples:")
    print()
    
    for i, (script, title, desc) in enumerate(examples, 1):
        print(f"{i}. {title}")
        print(f"   📝 {desc}")
        print(f"   📄 {script}")
        print()
    
    print("0. Run all examples")
    print("q. Quit")
    print()
    
    return examples

def main():
    """Main function"""
    examples = show_menu()
    
    while True:
        try:
            choice = input("👉 Choose example to run (number/0/q): ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            
            if choice == '0':
                # Run all examples
                print("\n🚀 Running all examples...")
                results = []
                
                for script, title, desc in examples:
                    print(f"\n⏳ Preparing to run: {title}")
                    input("Press Enter to continue or Ctrl+C to skip...")
                    
                    success = run_example(script, title)
                    results.append((title, success))
                    
                    if not success:
                        print(f"⚠️ {title} failed. Continue with next example? (y/n)")
                        if input().strip().lower() != 'y':
                            break
                
                # Summary
                print(f"\n{'='*60}")
                print("📊 SUMMARY OF ALL EXAMPLES")
                print(f"{'='*60}")
                
                for title, success in results:
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(f"{title:<30} : {status}")
                
                passed = sum(1 for _, success in results if success)
                total = len(results)
                print(f"\nResults: {passed}/{total} examples passed")
                
                break
            
            try:
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(examples):
                    script, title, desc = examples[choice_idx]
                    
                    print(f"\n📋 {title}")
                    print(f"📝 {desc}")
                    print(f"📄 {script}")
                    print()
                    
                    confirm = input("▶️ Run this example? (y/n): ").strip().lower()
                    
                    if confirm == 'y':
                        success = run_example(script, title)
                        
                        if success:
                            print(f"\n🎉 {title} completed successfully!")
                        else:
                            print(f"\n💥 {title} encountered errors")
                        
                        input("\nPress Enter to continue...")
                    
                    # Show menu again
                    print("\n" + "="*50)
                    examples = show_menu()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()