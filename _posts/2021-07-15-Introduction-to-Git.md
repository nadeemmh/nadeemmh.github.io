---
title: "Introduction to Git"
categories:
  - education
tags:
  - git
  - github
  - educational
---

This article focuses on understanding and using Git as a beginner to version control systems. 

# What is Git?

Git is an Open Source Distributed Version Control System. Git is used to store content such as code and allows for changes to be made to files, so that there is a record of what has been done, and the user can revert to specific versions should they ever need to. 

## Key Words: 
- Control System: This means that Git is a content tracker. So Git can be used to store content — it is mostly used to store code due to the other features it provides.
- Version Control System: The code which is stored in Git keeps changing as more code is added, and many developers can add code in parallel. So Version Control System helps in handling this by maintaining a history of what changes have been made. Git also provides features like branches and merges, which will be covered later.
- Distributed Version Control System: Git has a remote repository which is stored in a server and a local repository which is stored in the computer of each developer. This means that the code is not only stored in a central server, but the full copy of the code is also present in each user's computer. Git is a Distributed Version Control System since the code is present in every developer’s computer.

## Why Version Control Systems like Git are needed:
- Team projects generally have multiple developers working in parallel. So a version control system like Git is needed to ensure there are no conflicts in the code between the developers.
- Requirements in such projects change often. So a version control system allows developers to revert and go back to an older version of the code.
- Several projects which are being run in parallel involve the same codebase. In such a case, the concept of branching in Git is very important.

# Using Git

## Downloading Git:
This link has details on how to install Git in multiple operating systems:

https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Verify if Git is installed by using the following command in the command prompt:

```r
$ git --version
```

## Setting up Git:

**Help:**

```r
$ git help <command>

$ git <command> —help 
```

**Set config values:** 

```r
$ git config user.name "First Last"

$ git config use.email "emailaddress@email.com"

$ git config —list
``` 
## Working on a Project:
There are two main ways of working on a project in Git:

### 1. Working with a new project:

**Initialising a repository and adding a project onto Git**
Locate the local directory and use 'cd' to the work in the directory. Then to begin tracking the project with git use the following command:

```r
$ git init 
```

This initialises an empty git repository and creates a ".git" directory within the local directory which verifies that a repository has been created. This .git directory contains everything that is related to the repository.

**Removing a project from git:**

To stop tracking the project with git, the following command can be used within the directory in the terminal: 

```r
$ rm -rf .git 
```

This project is now no longer being tracked with git.

### Making a commit

**Before commit:**

The following command can be used to track the progress of a commit at any step of the process:

```r
$ git status
```

This will show untracked and tracked files.

**Ignoring specific files:**

To ignore files from being added onto the repository and being viewed by others, a .gitignore text file can be made to specify which files need to be ignored. 

1. Create a .gitignore file using the following command:
    ```r
    $ touch .gitignore 
    ```
2. Open this newly created file in a text editor and add the files that are not needed.
3. Now rerun the status command and observe that the ignored files are no longer visible in the list of untracked files and have now been replaced by the .gitignore file. This now needs to be committed in order for the changes to be made.

> ![layers](https://www.google.com/url?sa=i&url=https%3A%2F%2Fgit-scm.com%2Fbook%2Fen%2Fv2%2FGetting-Started-What-is-Git%253F&psig=AOvVaw1EXsNVkHnOMgefDhN5xxac&ust=1626446611899000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCLiRtfen5fECFQAAAAAdAAAAABAJ)

**Understanding the committing process:**

With git there are three states that need to be considered: 

1. Working Directory - untracked and modified files.
2. Staging Area - where we organise what we want to commit to the repo. The reason for the staging area is so that we can choose exactly what to commit. This is especially useful if there has been lots of work done in multiple files, then individual files can be staged and committed in chunks. This allows for more detailed commits (with the messages). 
3. Committed files - tracked and unmodified files.

**Adding files to the staging area:**

```r
$ git add -A (adds all untracked files)

$ git add .file (adds specific file)

$ git status (shows updated tracking and which files have been moved to the staging area. Important step!)
```

**Removing files from the staging area:**

```r
$ git reset .file (removed specific files from staging area)

$ git status 
```

**Committing selected files:** 

```r
1. $ git add .file
2. $ git commit -m "message for the commit"
3. $ git status (should say "nothing to commit, working directory clean")
4. $ git log (shows the commit that was just made which gives the hash number for commit and some details of the commit)
```

### 2. Working with an existing project:
