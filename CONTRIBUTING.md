# 🤝 Contributing — Team Guide

This guide is for the team members working on this project together.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/roommate-compatibility.git
cd roommate-compatibility
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Work Without Overwriting Each Other

### Always pull before you start working
```bash
git pull origin main
```
This gets the latest changes from your teammates before you start.

### Create a branch for your work
```bash
git checkout -b your-name/what-youre-doing
# Example:
git checkout -b kirti/data-cleaning
git checkout -b sanvitha/feature-encoding
```

### Commit your changes
```bash
git add .
git commit -m "Short description of what you did"
```

### Push your branch
```bash
git push origin your-name/what-youre-doing
```

### Open a Pull Request on GitHub
- Go to the repo on GitHub
- Click **"Compare & pull request"**
- Write a short description of what you changed
- Tag a teammate to review
- Merge when approved

---

## Who Works on What

| Area | Owner |
|---|---|
| Data collection & form | All three |
| Tokenisation & cleaning | *(assign)* |
| Feature encoding | *(assign)* |
| Model experiments | *(assign)* |
| Documentation | *(assign)* |

---

## Rules

- **Never push directly to `main`** — always use a branch and pull request
- **Always pull before starting** — avoids merge conflicts
- **Commit often** — small commits are easier to review and undo
- **Write clear commit messages** — *"fix encoding bug in sleep column"* not *"fix stuff"*
- **Don't commit raw data** — keep response data out of the repo for privacy
