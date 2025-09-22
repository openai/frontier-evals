#!/usr/bin/env python3
"""
Test script for Gemini 2.5 Pro integration with PaperBench
Usage: python test_gemini_integration.py
"""

import asyncio
import os
from pathlib import Path

# Add the preparedness_turn_completer to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "preparedness_turn_completer"))

from preparedness_turn_completer.google_completions_turn_completer import GoogleCompletionsTurnCompleter
from paperbench.nano.structs import JudgeConfig


async def test_google_completer():
    """Test the GoogleCompletionsTurnCompleter directly"""
    print("🧪 Testing GoogleCompletionsTurnCompleter...")

    # Check if Google API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found. Please set it to test Gemini integration.")
        return False

    try:
        # Create the completer
        completer = GoogleCompletionsTurnCompleter(
            model="gemini-2.5-pro",
            api_key=api_key,
            temperature=0.1,
        )
        print(f"✅ GoogleCompletionsTurnCompleter created successfully")
        print(f"   - Model: {completer.model}")
        print(f"   - Context window: {completer.n_ctx:,} tokens")
        print(f"   - Encoding: {completer.encoding_name}")

        # Test a simple completion
        test_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Gemini 2.5 Pro!' and nothing else."}
        ]

        print("\n🔄 Testing completion...")
        response = await completer.async_completion(test_conversation)

        print(f"✅ Completion successful!")
        print(f"   - Response: {response.output_messages[0].content}")
        print(f"   - Prompt tokens: {response.usage.prompt_tokens if response.usage else 'N/A'}")
        print(f"   - Completion tokens: {response.usage.completion_tokens if response.usage else 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ Error testing GoogleCompletionsTurnCompleter: {e}")
        return False


def test_judge_config():
    """Test the JudgeConfig with Gemini support"""
    print("\n🧪 Testing JudgeConfig with Gemini...")

    # Check if Google API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found. Skipping JudgeConfig test.")
        return False

    try:
        # Test with OpenAI (default)
        config_openai = JudgeConfig()
        print(f"✅ Default JudgeConfig (OpenAI): {type(config_openai.completer_config).__name__}")

        # Test with Gemini
        config_gemini = JudgeConfig(
            use_gemini=True,
            gemini_api_key=api_key
        )
        print(f"✅ Gemini JudgeConfig: {type(config_gemini.completer_config).__name__}")
        print(f"   - Model: {config_gemini.completer_config.model}")
        print(f"   - API Key set: {'Yes' if config_gemini.completer_config.api_key else 'No'}")
        print(f"   - Docker env has GOOGLE_API_KEY: {'GOOGLE_API_KEY' in config_gemini.cluster_config.environment}")

        return True

    except Exception as e:
        print(f"❌ Error testing JudgeConfig: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Testing Gemini 2.5 Pro Integration for PaperBench\n")

    # Test 1: GoogleCompletionsTurnCompleter
    test1_success = await test_google_completer()

    # Test 2: JudgeConfig
    test2_success = test_judge_config()

    # Summary
    print(f"\n📊 Test Results:")
    print(f"   GoogleCompletionsTurnCompleter: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   JudgeConfig Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! Gemini 2.5 Pro integration is working.")
        print(f"\n📋 Usage Instructions:")
        print(f"   1. Set GOOGLE_API_KEY environment variable")
        print(f"   2. Run PaperBench with: paperbench.judge.use_gemini=true")
        print(f"   3. Example command:")
        print(f"      uv run python -m paperbench.nano.entrypoint \\")
        print(f"          paperbench.judge.use_gemini=true \\")
        print(f"          paperbench.judge.gemini_api_key=$GOOGLE_API_KEY \\")
        print(f"          # ... other parameters")
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")

    return test1_success and test2_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
