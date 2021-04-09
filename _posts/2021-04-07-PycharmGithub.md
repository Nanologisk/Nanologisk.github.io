---
layout: article
title: Setting up in Github in Pycharm
tags: Pycharm, Github
modify_date: 2021-04-05
aside:
  toc: true
---

It is working now.  :ghost:

<!--more-->
### 1. Setting up in Github in Pycharm
- **Settings/Preferences** dialog `Ctrl+Alt+S`, select **Version Control
  | GitHub** in the left pane
- Click **+**
- In the dialog box, specify GitHub server URL (github.com).
  + Obtain a new token, enter login information and password
  + Or **Add account** and **Sign up for GitHub** (if creating a new GH
    account).
- More information can be found in
  [Pycharm's page](https://www.jetbrains.com/help/pycharm/github.html)

### 2. Clone resipotory
- **VCS** -> **Git | Clone**
- **Clone Repository**
  + URL: the GH address will show if already logged in
  + Directory: the local directory
  + If not log in, click on **Log in to GitHub**, fill in:
    * **Server**: github.com, **Login**: email-address, **Password**.

### 3. Change branches if necessary
- Right click the folder that need to change branches
- **Git | Branches**
  + The pop-up window: Git Branches: **Local Branches**: master,
    **Remote Branches**: origin/master, orgin/...
  + Check on the GH-site which branch the folder belongs to
  + Choose the branch -> **Checkout**

###  4. Manage project hosted on GitHub- Configure and synchronise
- Edit a file (e.g. test.md file)
- Right click the file, **Git | Commit Directory**
- The pop-up window will show changes (bottom left vs bottom right
  versions), as well as allowing for **commit message** to note which
  changes that have been done.
- Right click on the file in the pop-up window will allow to **Show
  Diff** `Ctrl+D`, which will show the changes modified. The same window
  shows also alternative to revert changes: **Revert** `Ctrl+Alt+Z`.
  Here is the [screenshot](../docs/assets/images/image.png).
- If everything looks ok, click **Commit** in the **Commit changes**
  pop-up window.
-

