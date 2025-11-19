# Security Guide

## 🔒 Security Features

This satellite terrain classification system has been hardened with comprehensive security features to protect against common vulnerabilities and ensure safe operation in production environments.

### Security Layers

1. **Input Validation & Sanitization**
2. **Authentication & Authorization**
3. **Rate Limiting**
4. **CORS Protection**
5. **Resource Management**
6. **Secrets Management**
7. **Error Handling**
8. **Security Logging**

---

## 📋 Security Checklist

Before deploying to production, ensure:

- [ ] Changed `SECRET_KEY` from default value
- [ ] Configured allowed CORS origins (not `*`)
- [ ] Enabled authentication (`ENABLE_AUTH=true`)
- [ ] Set up API keys for users
- [ ] Configured rate limiting appropriately
- [ ] Set file upload size limits
- [ ] Reviewed and configured allowed file extensions
- [ ] Set up HTTPS/TLS (use reverse proxy like Nginx)
- [ ] Configured firewall rules
- [ ] Set up monitoring and alerting
- [ ] Reviewed log retention policies
- [ ] Tested backup and recovery procedures

---

## 🛡️ Input Validation

All user inputs are validated to prevent injection attacks and ensure data integrity.

### File Upload Security

```python
from security.validation import FileValidator

# Validate image file
validated_path = FileValidator.validate_image_file(uploaded_file)
```

**Protections:**
- File size limits (default: 100MB)
- Extension whitelist (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`)
- Content-type verification
- Image corruption detection
- Path traversal prevention

### String Input Security

```python
from security.validation import InputValidator

# Validate and sanitize string
clean_input = InputValidator.validate_string(
    user_input,
    min_length=1,
    max_length=1000
)
```

**Protections:**
- Length limits
- Pattern matching
- Character whitelist
- HTML escaping
- SQL injection prevention
- Null byte removal

---

## 🔐 Authentication

### API Key Authentication

API keys provide simple, secure authentication.

**Setup:**

```python
from security.authentication import APIKeyAuth

auth = APIKeyAuth()
api_key = auth.generate_api_key("user@example.com")
print(f"API Key: {api_key}")
```

**Usage:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/predict
```

**Environment Variables:**

```bash
export ENABLE_AUTH=true
export API_KEY_HEADER="X-API-Key"
```

### JWT Token Authentication

For advanced use cases, JWT tokens are available.

```python
from security.authentication import JWTAuth

jwt_auth = JWTAuth()
token = jwt_auth.create_access_token({"user_id": "123"})
```

---

## ⏱️ Rate Limiting

Prevents abuse by limiting request rates per user.

### Default Limits

- **Per Minute**: 60 requests
- **Per Hour**: 1,000 requests
- **Per Day**: 10,000 requests

### Configuration

```bash
export RATE_LIMIT_PER_MINUTE=60
export RATE_LIMIT_PER_HOUR=1000
export RATE_LIMIT_PER_DAY=10000
```

### Custom Limits

```python
from security.authentication import RateLimiter

limiter = RateLimiter(
    requests_per_minute=100,
    requests_per_hour=5000,
    requests_per_day=50000
)
```

---

## 🌐 CORS Configuration

Cross-Origin Resource Sharing is configured to allow only trusted origins.

### Production Configuration

```bash
# Only allow specific origins
export ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# Restrict allowed hosts
export ALLOWED_HOSTS="yourdomain.com,api.yourdomain.com"
```

**⚠️ WARNING**: Never use `allow_origins=["*"]` in production!

---

## 🔑 Secrets Management

Sensitive credentials are encrypted and stored securely.

### Setup

```python
from config.secrets import SecretsManager

# Initialize with encryption key
secrets = SecretsManager()

# Store secrets
secrets.set("DATABASE_PASSWORD", "super-secret-password")
secrets.set("API_KEY", "your-api-key")
secrets.save()

# Retrieve secrets
db_password = secrets.get("DATABASE_PASSWORD")
```

### Environment Variables

```bash
# Set encryption key
export SECRETS_KEY="your-base64-encoded-key"
```

**⚠️ IMPORTANT**: Store the `SECRETS_KEY` securely (e.g., AWS Secrets Manager, HashiCorp Vault)

---

## 🛠️ Resource Management

Prevents resource exhaustion attacks.

### Memory Limits

```python
from utils.resource_manager import get_resource_manager

manager = get_resource_manager()

# Check memory usage
memory_stats = manager.check_memory()
print(f"Memory usage: {memory_stats['percent']}%")

# Enforce memory limit
manager.enforce_memory_limit()
```

### GPU Management

```python
# Configure GPU memory growth
manager = ResourceManager(
    enable_gpu_growth=True,
    gpu_memory_fraction=0.9
)
```

### File Handle Management

```python
from utils.resource_manager import FileHandleManager

file_manager = FileHandleManager(max_open_files=100)

with file_manager.open_file("data.txt") as f:
    data = f.read()
# File automatically closed
```

---

## 🚨 Error Handling

All errors are caught and logged securely without exposing sensitive information.

### Secure Error Messages

**❌ BAD** (exposes internals):
```json
{
  "error": "Database connection failed: postgres://user:password@localhost/db"
}
```

**✅ GOOD** (safe for users):
```json
{
  "error": "Internal server error",
  "detail": "Service temporarily unavailable"
}
```

### Logging Errors

All errors are logged with full details for debugging:

```python
try:
    risky_operation()
except Exception as e:
    logger.exception("Operation failed")  # Logs full traceback
    raise HTTPException(status_code=500, detail="Internal error")  # Safe user message
```

---

## 📊 Security Monitoring

### Metrics

Prometheus metrics track security events:

- `auth_failures_total` - Failed authentication attempts
- `terrain_prediction_errors_total` - API errors by type
- `rate_limit_exceeded_total` - Rate limit violations

### Log Analysis

Security-relevant events are logged:

```log
2025-11-19 12:34:56 | WARNING | auth:validate - Invalid API key attempted: abc12345...
2025-11-19 12:35:01 | WARNING | rate_limiter:check - User user@example.com exceeded rate limit
2025-11-19 12:35:10 | ERROR | validation:validate_file - File too large: 150MB (max: 100MB)
```

---

## 🐳 Docker Security

### Security Features

1. **Non-root user**: Application runs as `appuser` (UID 1000)
2. **Read-only filesystem**: Container filesystem is read-only
3. **No new privileges**: `security_opt: no-new-privileges:true`
4. **Resource limits**: CPU and memory limits enforced
5. **Minimal base image**: Uses slim Python image
6. **Multi-stage build**: Reduces attack surface

### Running Securely

```bash
docker run \
  --security-opt=no-new-privileges:true \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/logs \
  -e ENABLE_AUTH=true \
  -e SECRET_KEY=your-secret-key \
  satellite-terrain-classifier:latest
```

---

## 🔍 Security Testing

### Running Security Tests

```bash
# Run all tests including security
pytest tests/ -v

# Run security-specific tests
pytest tests/test_validation.py tests/test_sanitization.py -v
```

### Vulnerability Scanning

```bash
# Scan Python dependencies
safety check -r requirements.txt

# Scan Docker image
docker scan satellite-terrain-classifier:latest
```

---

## 📝 Security Best Practices

### Development

1. **Never commit secrets** - Use `.env` files (gitignored)
2. **Use type hints** - Helps catch errors early
3. **Validate all inputs** - Never trust user data
4. **Sanitize outputs** - Prevent XSS attacks
5. **Use parameterized queries** - Prevent SQL injection
6. **Keep dependencies updated** - Patch vulnerabilities

### Deployment

1. **Use HTTPS** - Encrypt data in transit
2. **Set strong secrets** - Use `secrets.token_urlsafe(32)`
3. **Enable authentication** - Don't expose APIs publicly
4. **Configure rate limiting** - Prevent abuse
5. **Monitor logs** - Detect attacks early
6. **Regular backups** - Prepare for disasters
7. **Least privilege** - Only grant necessary permissions

### Network Security

```nginx
# Example Nginx reverse proxy with security headers
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy to API
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
}
```

---

## 🚀 Production Deployment Checklist

### Pre-Deployment

- [ ] Security audit completed
- [ ] All tests passing
- [ ] Dependencies scanned for vulnerabilities
- [ ] Secrets properly configured
- [ ] Environment variables set
- [ ] HTTPS/TLS configured
- [ ] Firewall rules configured
- [ ] Monitoring set up
- [ ] Backup strategy in place

### Post-Deployment

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Alerts configured
- [ ] Access controls verified
- [ ] Rate limiting tested
- [ ] Disaster recovery tested

---

## 📞 Security Contact

If you discover a security vulnerability, please email: security@yourdomain.com

**DO NOT** create public GitHub issues for security vulnerabilities.

### Responsible Disclosure

We appreciate security researchers who:

1. Give us reasonable time to fix issues (90 days)
2. Don't exploit vulnerabilities
3. Don't access user data
4. Report responsibly

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://www.sans.org/top25-software-errors/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Last Updated**: 2025-11-19
**Security Level**: Production-Ready ✅
