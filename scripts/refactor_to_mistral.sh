#!/bin/bash
# Script to refactor from Llama to Mistral across the codebase

# Exit on error
set -e

echo "🔄 Starting Llama → Mistral refactoring..."

# Files to process (excluding git, pycache, etc.)
FILES=$(find . -type f \( -name "*.py" -o -name "*.md" \) \
    ! -path "./.git/*" \
    ! -path "./aifs-single-1.0/*" \
    ! -path "./__pycache__/*" \
    ! -path "./*.egg-info/*" \
    ! -path "./.pytest_cache/*")

# Create backup
echo "📦 Creating backup..."
tar -czf llama_to_mistral_backup_$(date +%Y%m%d_%H%M%S).tar.gz $FILES

# Replacement patterns
echo "🔧 Applying replacements..."

for file in $FILES; do
    # Skip if file doesn't exist or is binary
    if [ ! -f "$file" ] || file "$file" | grep -q "binary"; then
        continue
    fi
    
    # Model name replacements
    sed -i '' 's/meta-llama\/Meta-Llama-3-8B/mistralai\/Mistral-7B-Instruct-v0.3/g' "$file"
    sed -i '' 's/Meta-Llama-3-8B/Mistral-7B-Instruct-v0.3/g' "$file"
    sed -i '' 's/Llama-3-8B/Mistral-7B-Instruct/g' "$file"
    sed -i '' 's/Llama 3-8B/Mistral-7B-Instruct/g' "$file"
    sed -i '' 's/Llama 3 8B/Mistral-7B-Instruct/g' "$file"
    sed -i '' 's/Llama-3/Mistral-7B/g' "$file"
    sed -i '' 's/llama-3/mistral-7b/g' "$file"
    sed -i '' 's/llama3/mistral7b/g' "$file"
    
    # Class/variable name replacements
    sed -i '' 's/AIFSLlamaFusionModel/AIFSMistralFusionModel/g' "$file"
    sed -i '' 's/aifs_llama/aifs_mistral/g' "$file"
    sed -i '' 's/AIFS-Llama/AIFS-Mistral/g' "$file"
    sed -i '' 's/AIFS-LLaMA/AIFS-Mistral/g' "$file"
    sed -i '' 's/llama_tokens/mistral_tokens/g' "$file"
    sed -i '' 's/LLaMA/Mistral/g' "$file"
    
    # Generic replacements (be careful with these)
    sed -i '' 's/\bLlama\b/Mistral/g' "$file"
    sed -i '' 's/\bllama\b/mistral/g' "$file"
    sed -i '' 's/LLAMA/MISTRAL/g' "$file"
    
    # File-specific patterns
    sed -i '' 's/test_aifs_llama/test_aifs_mistral/g' "$file"
    sed -i '' 's/test_real_llama/test_real_mistral/g' "$file"
    sed -i '' 's/test_cpu_llama/test_cpu_mistral/g' "$file"
    
    # Parameter count updates
    sed -i '' 's/8\.03B/7.25B/g' "$file"
    sed -i '' 's/8B parameters/7.25B parameters/g' "$file"
    sed -i '' 's/(8B)/(7B)/g' "$file"
done

echo "✅ Text replacements complete!"
echo "📝 Note: Manual review and file renaming still required"
echo "🔍 Check backup file if needed to revert"
