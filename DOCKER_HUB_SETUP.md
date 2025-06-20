# Docker Hub CI/CD Setup Guide

This guide will help you set up automated CI/CD pipelines to build and push Docker images to Docker Hub using GitHub Actions.

## 🚀 Quick Start

### 1. Create Docker Hub Account
If you don't have a Docker Hub account, create one at [hub.docker.com](https://hub.docker.com/)

### 2. Create Docker Hub Access Token
1. Log in to Docker Hub
2. Go to **Account Settings** → **Security**
3. Click **"New Access Token"**
4. Give it a descriptive name (e.g., "GitHub Actions CI/CD")
5. **Copy the token** - you won't be able to see it again!

### 3. Add GitHub Secrets
1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add these secrets:
   - **Name**: `DOCKERHUB_USERNAME`
   - **Value**: Your Docker Hub username
5. Add another secret:
   - **Name**: `DOCKERHUB_TOKEN`
   - **Value**: The access token you created in step 2

### 4. Test the Setup
1. Go to **Actions** tab in your repository
2. Find **"Test Docker Hub Setup"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. Check if the workflow completes successfully

## 📋 Available Workflows

### 1. Docker Hub CI/CD (`docker-hub-ci.yml`)
**Triggers**: Push to main/master, Pull requests, Tags
**Features**:
- Runs tests before building
- Builds multi-platform images (AMD64, ARM64)
- Pushes to Docker Hub with appropriate tags
- Security scanning with Trivy
- Success/failure notifications

### 2. Docker Hub Release (`release-docker-hub.yml`)
**Triggers**: Tags (v*)
**Features**:
- Creates GitHub releases
- Pushes versioned images to Docker Hub
- Uploads docker-compose.yaml as release asset
- Automatic release notes

### 3. Security Scan (`security-scan.yml`)
**Triggers**: Weekly schedule, Manual
**Features**:
- Vulnerability scanning of latest image
- Uploads results to GitHub Security tab
- Alerts on critical vulnerabilities

### 4. Test Setup (`test-docker-hub.yml`)
**Triggers**: Manual only
**Features**:
- Tests Docker Hub authentication
- Verifies image building
- No image pushing (safe for testing)

## 🏷️ Image Tagging Strategy

The workflows use the following tagging strategy:

- **`latest`**: Latest stable build from main/master branch
- **`main`**: Latest build from main branch
- **`v1.0.0`**: Specific version tags
- **`v1.0`**: Major.minor version tags
- **`main-abc123`**: Branch name with commit SHA

## 🔧 Configuration

### Customizing Image Names
Edit the workflow files to change the image name:

```yaml
env:
  REGISTRY: docker.io
  IMAGE_NAME: your-custom-name  # Change this
```

### Adding More Platforms
Modify the `platforms` parameter in build steps:

```yaml
platforms: linux/amd64,linux/arm64,linux/arm/v7
```

### Custom Security Scan Schedule
Edit the cron expression in `security-scan.yml`:

```yaml
schedule:
  - cron: '0 9 * * 1'  # Every Monday at 9 AM UTC
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Authentication Failed
**Error**: `unauthorized: authentication required`
**Solution**: 
- Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets
- Ensure the token has write permissions
- Check if the token is expired

#### 2. Repository Not Found
**Error**: `repository does not exist`
**Solution**:
- Create the repository on Docker Hub first
- Ensure the repository name matches your GitHub repo name
- Check if the repository is public or you have access

#### 3. Build Failures
**Error**: `failed to build image`
**Solution**:
- Check the Dockerfile for syntax errors
- Verify all required files are present
- Check if the base image exists

#### 4. Permission Denied
**Error**: `permission denied`
**Solution**:
- Ensure the Docker Hub token has the correct permissions
- Check if your Docker Hub account has the necessary limits

### Debugging Steps

1. **Run the test workflow** to verify basic setup
2. **Check workflow logs** for detailed error messages
3. **Verify secrets** are correctly set
4. **Test Docker Hub login** manually:
   ```bash
   docker login -u your-username -p your-token
   ```

## 📊 Monitoring

### Workflow Status
- Monitor workflow runs in the **Actions** tab
- Set up notifications for workflow failures
- Check the **Security** tab for vulnerability reports

### Docker Hub Metrics
- View image pulls in Docker Hub dashboard
- Monitor storage usage
- Check build history

## 🔒 Security Best Practices

1. **Use Access Tokens**: Never use your Docker Hub password
2. **Token Permissions**: Only grant necessary permissions
3. **Regular Rotation**: Rotate access tokens periodically
4. **Security Scanning**: Enable automated vulnerability scanning
5. **Private Repositories**: Use private repos for sensitive images

## 📈 Advanced Features

### Conditional Builds
Add conditions to only build on specific events:

```yaml
if: github.event_name != 'pull_request'
```

### Custom Build Arguments
Pass build arguments to Docker:

```yaml
build-args: |
  BUILD_VERSION=${{ github.sha }}
  BUILD_DATE=${{ github.event.head_commit.timestamp }}
```

### Multi-stage Builds
Optimize your Dockerfile for smaller images:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
# ... build steps

# Runtime stage
FROM python:3.11-slim
# ... copy from builder
```

## 🎯 Next Steps

1. **Test the setup** using the test workflow
2. **Push to main branch** to trigger the first build
3. **Create a release** by pushing a tag (e.g., `v1.0.0`)
4. **Monitor the pipelines** and adjust as needed
5. **Set up notifications** for build status

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review GitHub Actions documentation
3. Check Docker Hub documentation
4. Open an issue in this repository

---

**Happy CI/CDing! 🚀** 