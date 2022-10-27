# 👨‍💻 Contributing

Thank you for your interest in contributing to MAKit! From commenting to 
reviewing and sending PR requests, all contributions are welcome. 

- [Developer Environment](#developer-environment)
  - [Setup](#setup)
  - [Coding Style](#coding-style)
- [Testing](#testing)
  - [CI](#ci)
  - [Local](#local)
- [Git Workflow](#git-workflow)
- [Repo Admins and Code Owners](#repo-admins-and-code-owners)
  - [Admin](#admin)
  - [Code Owner](#code-owner)
- [Comit Message](#commit-message)
  - [Examples](#examples)
    - [Commit Message with Body](#commit-message-with-body)
    - [Commit Message with Scope](#commit-message-with-scope)
    - [Commit Message with both ! and BREAKING CHANGE Footer](
      #commit-message-with-both--and-breaking-change-footer)
- [Versioning](#versioning)
- [Release on GitHub](#release-on-github)
  - [Before you Start](#before-you-start)
  - [Step 1. Update the Changelog](#step-1-update-the-changelog)
  - [Step 2. Create a Pull Request](#step-2-create-a-pull-request)
  - [Step 3. Review and Merge the Pull Request](
    #step-3-review-and-merge-the-pull-request)
  - [Step 4. Create a Github release "X.Y.Z" from "main"](
    #step-4-create-a-github-release-xyz-from-main)
    
________________________________________________________________________________

## Developer Environment

### Setup

Install XCode with command line tools.

### Coding Style

Go to XCode preferences -> Text editing tab -> Page guide at column: 80.

## Testing

### CI 

The [unit tests](../Architecture/MAKitTests.md) 
are run after each push on the repository. 

The [integration tests](../Architecture/MATorchTests.md) 
are not run systematically, 
neither are the [examples](../Architecture/MAExamples.md).

Once the PR is "ready to review", please trigger the workflows on GitHub 
to ensure these additional tests have completed. 

### Local 

Testing the [unit tests](../Architecture/MAKitTests.md) 
on XCode is straight forward.

Testing the [integration tests](../Architecture/MATorchTests.md) 
and the [examples](../Architecture/MAExamples.md) require an additional setup. 
Follow the previous links to know more about.

## Git Workflow

- Git workflow: we use the Github Flow, with the modification. 
  Tags are used to indicate which commit can be used in production.

- The Main branch is always protected, 
  all updates to this branch arrive via pull requests (PRs).

- The name of the Main branch is `main`.

- Each PR :

  - Must be reviewed by at least one maintainer of the repo, 
    implemented as code owners in GitHub (see below).

  - Should be as small and focused as possible.

  - Should implement changes that are backward compatible as much as possible.

  - Should provide the context of the changes and document what it solves.

- Push often changes to Github; at least on a daily basis 
  even if the PR is not yet ready for accepting reviews. 
  In this case, use the DRAFT status.

- TODOs can be left in the code if another/others PR deals with them. 
  In that case, add the link to this/those PR in the description. 
  If no PR deals with them, 
  it is better to use Github issues for small changes / improvements.

- In case of a new feature or bug fix, 
  unit tests should be added to cover all the use cases. 
  They should help the reviewers to understand the PR.
  If such tests already exist, they should be mentioned in the description.

- Each important PR should update the CHANGELOG 
  (format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)), 
  or any other documentation (docstring, contributing, readme, ...)

- Squash before merge.

- Rebase your PR as often as possible, 
  if using `push force` then do it with lease and be careful about it 
  if several people work on the PR.

## Repo Admins and Code Owners

The key principle that is applied to grant roles to users is:
making sure workflow is smooth while limiting the number of people 
with extended rights for security purposes.

We detail below the responsibilities of each role and the processes to update 
the list of people with these roles.

#### Admin

Person who has [admin right](
https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization) 
for a repo on GitHub.
"She/he has full access to the project, including sensitive and 
destructive actions like managing security or deleting a repository",
this is why it needs to be limited to a maximum of 3 people. 

#### Code Owner

**Person** who knows well the code base and who is able to judge whether 
this can be merged in main or not.
The code owners have the following responsibilities:

- review PR before it can be merged into the main branch.
- handle GitHub Issues lifecycle.

Code owners are added by existing code owners, 
everybody can suggest new code owners.

## Commit Message

The commit message must follow the 
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0) 
specification. 

Please also use mind the following 
[conventional emoji commits](
https://gist.github.com/parmentf/359667bf23e08a1bd8241fbf47ecdef0).

On top of the specification, we apply the following rules:

- recommended prefixes list: `fix`, `feat`, `build`, `chore`, `ci`, `docs`, 
  `style`, `refactor`, `perf` and `test`
- header line must be lower than 100 characters
- for breaking change, the commit must include 
  a footer with `BREAKING CHANGE:` and `!` after the type/scope

### Examples

#### Commit Message with Body

```
✨ feat: add options for auto batching in the CLI

Lorem ipsum dolor sit amet. Est soluta dolores rem itaque suscipit qui 
soluta porro 33 galisum rerum aut numquam voluptates qui Quis deserunt. 
```

#### Commit Message with Scope

```
🐛 fix(core): add unique constraint on tag name
```

#### Commit Message with both ! and BREAKING CHANGE Footer

```
🔨 refactor!: remove GPU variable

BREAKING CHANGE: deprecate GPU variable
```

## Versioning

- This project adheres to [Semantic Versioning](https://semver.org).
- To bump the release version, MAKit follows the rules below:
  - X is at zero (MAKit is in development phase)
  - Y is bumped for new features, architecture changes 
    and non backwards compatible fixes
  - Z is bumped when backwards compatible bug fixes are introduced

## Release on GitHub

Let say we want to make a release for the branch `release_1` containing the 
commits to merge into the `main` branch.

### Before you Start

Ensure all [end-to-end tests](#ci) pass on `release_1`.

### Step 1. Update the Changelog

1. Create a branch from `release_1`.

1. Edit the changelog to create a new section corresponding to the new release. 
   Use [git-cliff](https://github.com/orhun/git-cliff) for that.
  
1. Commit and push the changes.

1. Squash and merge the new branch into `release_1` with commit message 
   "🔧 chore: release X.Y.Z".

### Step 2. Create a Pull Request

### Step 3. Review and Merge the Pull Request

### Step 4. Create a GitHub release "X.Y.Z" from "main"
    
- GitHub > Releases > Draft new Release
  - **Choose a tag**: X.Y.Z (this will create the tag)
  - **Target**: main
  - **Release title**: X.Y.Z
  - **Describe this release:** formatted copy of the changelog.
