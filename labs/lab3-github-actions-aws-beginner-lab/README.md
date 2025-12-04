# GitHub Actions + AWS S3: ML Automation Lab

Automate machine learning model training and storage using GitHub Actions and AWS S3.

## What You'll Learn

- Connect GitHub Actions to AWS
- Automate ML model training
- Store models in S3 automatically

## Quick Setup

### 1. Create GitHub Repository
```bash
git clone https://github.com/AardwolfFizzler/MLOPS.git
cd MLOPS/labs/github-actions-aws-beginner-lab

# Copy to new location
cp -r . ~/my-aws-lab
cd ~/my-aws-lab
rm -rf .git

# Push to your new repo
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### 2. Setup AWS

**Create IAM User:**
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Create user → Attach `AmazonS3FullAccess` policy
3. Create access keys → Save Access Key ID and Secret Key

**Create S3 Bucket:**
1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Create bucket with a unique name
3. Note the bucket name and region

### 3. Add GitHub Secrets

Go to your repo **Settings** → **Secrets and variables** → **Actions**

Add these secrets:
- `AWS_ACCESS_KEY_ID` - Your AWS access key
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret key  
- `AWS_S3_BUCKET` - Your bucket name
- `AWS_REGION` - Your AWS region (e.g., `us-east-1`)
- `AWS_SESSION_TOKEN` - Only if using temporary credentials

### 4. Run Workflow

1. Go to **Actions** tab in GitHub
2. Click **"Train and save model to S3"**
3. Click **"Run workflow"**
4. Check your S3 bucket for the trained model!

## Files

- `train_and_save_model.py` - Trains RandomForest on Iris dataset, uploads to S3
- `requirements.txt` - Dependencies (pandas, scikit-learn, boto3)
- `.github/workflows/train-and-upload.yml` - GitHub Actions workflow

## How It Works

The workflow:
1. Runs daily at midnight (or manually)
2. Sets up Python environment
3. Trains a RandomForest model
4. Uploads model to S3 as `trained_models/model_[timestamp].joblib`

## Troubleshooting

**"Invalid security token"** → Credentials expired, generate new ones

**"Access Denied"** → Check IAM permissions and bucket name

**No workflow trigger** → Ensure file is on `main` branch

## Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [AWS S3 Docs](https://docs.aws.amazon.com/s3/)
- [Boto3 Docs](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
