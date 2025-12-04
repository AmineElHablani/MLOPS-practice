# Lab 1: GitHub Basics & Testing

This lab covers fundamental MLOps practices including virtual environments, GitHub repositories, Python development, testing with pytest and unittest, and GitHub Actions automation.

## Learning Objectives

- Create and manage Python virtual environments
- Set up GitHub repositories with proper structure
- Write and organize Python code
- Implement automated testing with pytest and unittest
- Automate workflows using GitHub Actions

## Prerequisites

- Python 3.8 or higher
- Git installed
- GitHub account
- Basic Python knowledge

## Lab Structure
```
lab1-github-basics/
├── data/              # Project data files
├── src/               # Source code
│   └── calculator.py  # Sample calculator functions
├── test/              # Test files
│   ├── test_pytest.py
│   └── test_unittest.py
├── workflows/         # GitHub Actions workflows
│   ├── pytest_action.yml
│   └── unittest_action.yml
├── requirements.txt
└── README.md
```

## Step 1: Creating a Virtual Environment

Virtual environments isolate project dependencies from the global Python environment.

### Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv lab_01

# Activate (Mac/Linux)
source lab_01/bin/activate

# Activate (Windows)
lab_01\Scripts\activate
```

After activation, you'll see `(lab_01)` in your terminal prompt.

## Step 2: Setting Up Your GitHub Repository

### Option 1: Use This Lab from MLOPS Repository
```bash
# Clone the MLOPS repository
git clone https://github.com/AardwolfFizzler/MLOPS.git
cd MLOPS/labs/lab1-github-basics
```

### Option 2: Create Your Own Repository

1. **Create New Repository on GitHub**
   - Go to [GitHub](https://github.com)
   - Click "+" → "New repository"
   - Name it (e.g., `github-basics-lab`)
   - Choose public or private
   - Initialize with README
   - Click "Create repository"

2. **Clone Your Repository**
```bash
git clone https://github.com/your-username/github-basics-lab.git
cd github-basics-lab
```

3. **Create Project Structure**
```bash
mkdir -p src data test
touch .gitignore
```

4. **Create .gitignore**
```bash
cat > .gitignore << 'EOL'
# Virtual Environment
lab_01/
venv/
env/

# Python
__pycache__/
*.py[cod]
*.so
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOL
```

## Step 3: Creating calculator.py

The `src/calculator.py` file contains basic arithmetic functions. Review the existing file or create it:
```python
def fun1(x, y):
    """Add two numbers"""
    return x + y

def fun2(x, y):
    """Subtract y from x"""
    return x - y

def fun3(x, y):
    """Multiply two numbers"""
    return x * y

def fun4(x, y):
    """Combine all operations"""
    return fun1(x, y) + fun2(x, y) + fun3(x, y)
```

### Push Your Code
```bash
git add .
git commit -m "Add calculator module"
git push origin main
```

## Step 4: Creating Tests

### Install Testing Libraries
```bash
pip install pytest
pip freeze > requirements.txt
```

### Pytest Tests

The `test/test_pytest.py` file contains pytest-style tests. Key features:

- Simple function-based tests
- Use `assert` statements for validation
- Automatic test discovery

Run pytest tests:
```bash
pytest test/test_pytest.py -v
```

### Unittest Tests

The `test/test_unittest.py` file contains unittest-style tests. Key features:

- Class-based test structure
- Inherits from `unittest.TestCase`
- Uses `self.assertEqual()` for assertions

Run unittest tests:
```bash
python -m unittest test.test_unittest
```

## Step 5: Implementing GitHub Actions

GitHub Actions automate testing whenever code is pushed or pull requests are created.

### Workflows Included

This lab includes two workflow files in the `workflows/` folder:

1. **pytest_action.yml** - Runs pytest tests automatically
2. **unittest_action.yml** - Runs unittest tests automatically

### Setting Up Workflows in Your Repository

If creating your own repository, move the workflows:
```bash
mkdir -p .github/workflows
cp workflows/pytest_action.yml .github/workflows/
cp workflows/unittest_action.yml .github/workflows/
```

### Push Workflows to GitHub
```bash
git add .github/
git commit -m "Add GitHub Actions workflows"
git push origin main
```

## Testing Your GitHub Actions

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. You should see both workflows running
4. Click on a workflow to see detailed logs
5. Verify all tests pass ✅

## Understanding the Workflows

### Pytest Workflow Features

- Triggers on push to `main` branch
- Runs on Ubuntu latest
- Installs dependencies from `requirements.txt`
- Generates JUnit XML test report
- Uploads test results as artifacts

### Unittest Workflow Features

- Triggers on push to `main` branch
- Runs on Ubuntu latest
- Executes unittest test suite
- Provides success/failure notifications

## Common Commands Reference
```bash
# Virtual Environment
python -m venv lab_01
source lab_01/bin/activate  # Mac/Linux
lab_01\Scripts\activate     # Windows
deactivate

# Git
git status
git add .
git commit -m "message"
git push origin main

# Testing
pytest test/ -v
python -m unittest test.test_unittest

# Dependencies
pip install -r requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

**Import errors in tests**: Check that test files have correct import paths

**GitHub Actions failing**: Verify `requirements.txt` includes all dependencies (especially pytest)

**Virtual environment issues**: Deactivate and recreate: `deactivate && python -m venv lab_01`

**Tests not discovered**: Ensure test files start with `test_` and test functions start with `test_`

## Next Steps

- Add more complex functions to `calculator.py`
- Write parametrized tests in pytest
- Add code coverage reporting
- Implement continuous deployment
- Explore more GitHub Actions features

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Unittest Documentation](https://docs.python.org/3/library/unittest.html)

## License

Educational purposes
