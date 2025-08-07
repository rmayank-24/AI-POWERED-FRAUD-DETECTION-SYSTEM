# GitHub Push Commands Guide

## 🚀 Complete Git Commands for Your Repository

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Add All Files (with .gitignore protection)
```bash
git add .
```

### Step 3: Check Status
```bash
git status
```

### Step 4: Initial Commit
```bash
git commit -m "Initial commit: AI-Powered Transaction Fraud Detection System"
```

### Step 5: Add Remote Repository
```bash
git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git
```

### Step 6: Push to Main Branch
```bash
git push -u origin main
```

## 🔄 Additional Commands

### Check Remote
```bash
git remote -v
```

### Force Push (if needed)
```bash
git push -f origin main
```

### Pull Latest Changes
```bash
git pull origin main
```

### Create New Branch
```bash
git checkout -b feature/new-feature
```

### Push New Branch
```bash
git push -u origin feature/new-feature
```

## 📋 Complete Workflow

```bash
# 1. Initialize
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "feat: AI-powered fraud detection system with ML models"

# 4. Add remote
git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git

# 5. Push
git push -u origin main
```

## 🎯 One-Line Commands

### Quick Setup
```bash
git init && git add . && git commit -m "Initial commit" && git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git && git push -u origin main
```

### With Branch
```bash
git init && git checkout -b main && git add . && git commit -m "Initial commit" && git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git && git push -u origin main
```

## 🛡️ Security Check

Before pushing, verify:
```bash
# Check .gitignore
cat .gitignore

# Check for sensitive files
find . -name "*.env" -o -name "*.key" -o -name "*.pem" -o -name "secrets*"
```

## 📊 Status Commands
```bash
# Check status
git status

# Check remote
git remote -v

# Check log
git log --oneline
```

## 🚀 Push with Tags
```bash
# Add tag
git tag -a v1.0.0 -m "Initial release"

# Push with tags
git push origin main --tags
```

## 🔄 Force Push (Use Carefully)
```bash
git push -f origin main
```

## 📞 Troubleshooting

### If remote already exists
```bash
git remote remove origin
git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git
```

### If branch doesn't exist
```bash
git branch -M main
git push -u origin main
```

### If authentication needed
```bash
# Use GitHub CLI
gh auth login
git push origin main
```

## 🎯 Complete Setup Script

Save as `setup_git.sh`:

```bash
#!/bin/bash
echo "Setting up Git repository..."
git init
git add .
git commit -m "Initial commit: AI-Powered Transaction Fraud Detection System"
git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git
git branch -M main
git push -u origin main
echo "Repository pushed successfully!"
```

## 🚀 Quick Start Commands

Run these in order:

```bash
# 1. Initialize
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "feat: AI-powered fraud detection system with ML models, Docker, and real-time dashboard"

# 4. Add remote
git remote add origin https://github.com/rmayank-24/AI-POWERED-FRAUD-DETECTION-SYSTEM.git

# 5. Push
git push -u origin main
```

## 📋 Verification Commands

```bash
# Check if everything is ready
git status
git remote -v
git log --oneline -5
```

## 🎯 Ready to Push!

Your project is now ready to be pushed to GitHub. Run the commands above in sequence to push your complete AI-powered fraud detection system to your repository.
