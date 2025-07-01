# Start new project in Git

## 1. Create a new GitHub repository

1. Log in.
2. clike the `+` symbol, -> "New repository"
3. Fill inn
    * Repository name,
    * Description
    * Visibility (not check README yet)
4. Create repository

## 2. Clone the repo to machine

1. Open terminal:
```bash
mkdir -p ~/Projects
cd ~/projects
```

2. Clone repo
```bash
git clone https://github.com/yourusername/my-learning-log.git
```

3. Go to the repo folder:
```bash
cd my-new-repo
```

## 3. Create new folders
Inside `~/Projects/my-new-repo`:
```bash
mkdir Linux Nvim Python Statistics 
```

Create new markdown file
```bash
echo "# Python Notes" > Python/README.md
echo "# Linux Notes" > Linux/README.md
echo "# Nvim Notes" > Linux/README.md
echo "# Statistics Notes" > Statistics/README.md
```

Copy over some old files
```bash
cp /path/to/file/file.md ~/Projects/my-new-repo
```

Or copy an entire folder:
```bash
cp -r [source_folder] [destination_folder]
```

or copy content of the folder:
```bash
cp -r ~/OldNotes/* ~Projects/my-new-rep/Python/
```

**Be careful: files with the same name may be overwritten without warning!**

## 4. State, commit and push changes

- Let Git know we will track these new files and folders
- Check git status
- Commit changes with a message/text
- Push the changes to GitHub

```bash 
git add .
git status
git commit -m "Add category folders with starter README.md files"
git push 
```

### If token (password) is expired 

Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Fine-grained tokens", or "Classic tokens" → Generate new token
3. Fill in:
    * Name: something like Linux Git Token
    * Expiration: choose 30/60/90 days or “No expiration”
    * Repository access: Select Only select repositories or All repositories
    * Permissions:
        * check Contents > Read and Write
        * check pages > Read and Write (for webpage)
4. Click Generate token
5. Copy the token and store it safely (you won’t see it again). Paste it when Git asks for password.

```less
Repository access:
[ ] No access (default)
[ ] All repositories
[✓] Only select repositories → (then pick your learning-log repo)

Permissions:
[✓] Contents: Read and Write
```

**Save the tolken at once or it will be gone...**

6. To avoid entering the token everytime, cache it:
```bash
git config --global credential.helper store
```

7. Push and paste the tolken. Git will remember it for future pushese
```bash
git push 
```
paste - done!


## 5. Get the latest changes on the PC

Direct to local repo folder
```bash
cd ~/Projects/thisporject
```

Then run
```bash 
git pull
```
This will download and merge changes from GitHub into local files.


------------------

## Summarise:

```bash
# pull latest changes 
git pull

# make changes

# commit changes
git add.
git commit -m "comment"

# push commit to Github
git push 

```
