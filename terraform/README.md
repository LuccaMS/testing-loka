## ⚠️ Required Setup

This project builds Python requirements locally. **You must install pip on WSL before running Terraform:**

```bash
sudo apt update && sudo apt install python3-pip -y
```

### Disclaimers

Sadly, localstack does not allow lambdas to have streaming responses, hence, we cannot do streaming of llm tokens.