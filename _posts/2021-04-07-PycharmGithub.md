---
layout: article
title: Setting up Github in Pycharm
tags: Pycharm Github
modify_date: 2021-04-05
aside:
  toc: true
---

Commit changes and edit files with Pycharm and synchronise with GitHub.
:ghost:

<!--more-->
### 1. Setting up Github in Pycharm
- **Settings/Preferences** dialog `Ctrl+Alt+S`, select **Version Control
  \| GitHub** in the left pane
- Click **+**
- In the dialog box, specify GitHub server URL (github.com).
  + Obtain a new token, enter login information and password
  + Or **Add account** and **Sign up for GitHub** (if creating a new GH
    account).
- More information can be found in
  [Pycharm's page](https://www.jetbrains.com/help/pycharm/github.html)

### 2. Clone repository
- In the menu bar, choose **VCS** if any, or click direct **Git \|
  Clone** if **VCS** does not show in the menu-bar.
- **Clone Repository**
  + URL: the GH address will show if already logged in
  + Directory: the local directory
  + If not log in, click on **Log in to GitHub**, fill in:
    * **Server**: github.com, **Login**: email-address, **Password**.

### 3. Change branches if necessary
- Right click the folder that need to change branches
- Choose **Git \| Branches** here.
  + The pop-up window: Git Branches: **Local Branches**: master,
    **Remote Branches**: origin/master, orgin/...
  + Check on the GH-site which branch the folder belongs to
  + Choose the branch -> **Checkout**

###  4. Manage project hosted on GitHub- Configure and synchronise
- Edit a file (e.g. test.md file)
- Right click the file, and choose **Git \| Commit Directory**.
- The pop-up window will show changes (bottom left vs bottom right
  versions), as well as allowing for **commit message** to note which
  changes that have been done.
- Right click on the file in the pop-up window will allow to **Show
  Diff** `Ctrl+D`, which will show the changes modified. The same window
  shows also alternative to revert changes: **Revert** `Ctrl+Alt+Z`.
  Here is the [screenshot](../docs/assets/images/image.png).
- If everything looks ok, click **Commit** in the **Commit changes**
  pop-up window.
- In order to synchronise changes, click on the **push** symbol (green
  arrow in the up-right corner).
- Right click the file, then click on **Git | Push** `Ctrl+Shift+K`.
  + This will show a new window with **Push commits**, and here we can
    right click again and double check for **Show diff**. Finally, click
    **Push** in the bottom-left corner to push the file to GH.
  + A **Push successful** or **Everything is up to date** window will
    pop up in the bottom right corner to inform the action is done.
- NB! Always **commit file** before **push**.

### 5. Update the local folder with GH
- Use the **pull** symbol (blue arrow in the up-right corner) to update
  project `Ctrl + T`.
- Right click on the folder, so choose **Git \| pull**.

### Useful links
[How to configure and use GitHub from Pycharm](https://www.youtube.com/watch?v=7sinNdn49Uk)

[Setting up GitHub for Pycharm](https://www.jetbrains.com/help/pycharm/github.html)
