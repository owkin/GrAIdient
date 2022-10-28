# 👨‍💻 Contributing

Thank you for your interest in contributing to MAKit! From commenting to 
reviewing and sending PR requests, all contributions are welcome. 

- [Developer Environment](#developer-environment)
  - [Setup Xcode](#setup-xcode)
  - [Setup Conda](#setup-conda)
  - [Coding Style](#coding-style)
- [Testing](#testing)
  - [CI](#ci)
  - [Local](#local)
- [Git Workflow](#git-workflow)
- [Repo Admins, Code Owners, Authors](#repo-admins-code-owners-authors)
  - [Admin](#admin)
  - [Code Owner](#code-owner)
  - [Author](#author)
- [Comit Message](#commit-message)
  - [With a Body](#with-a-body)
  - [With a Scope](#with-a-scope)
  - [With both ! and BREAKING CHANGE Footer](
    #with-both--and-breaking-change-footer)
- [Versioning](#versioning)
- [Release on GitHub](#release-on-github)
- [CHANGELOG template](#changelog-template)

## Developer Environment

### Setup Xcode

Install Xcode from 
[Apple developer website](https://developer.apple.com/download/) 
or the App Store on your Mac.

Then install Xcode Command Line Tools from the terminal: 

```
xcode-select --install
```

You can verify you have successfully installed Xcode Command Line Tools: 

```
xcode-select -p
```

### Setup Conda

Install anaconda from brew with: 

```
brew install --cask anaconda
```

### Coding Style

Go to Xcode preferences > Text editing tab > Page guide at column: 80.

## Testing

### CI 

The [unit tests](../Architecture/MAKitTests.md) 
are run after each push on the repository. 

The [integration tests](../Architecture/MATorchTests.md) 
are not run systematically, 
neither are the [examples](../Architecture/MAExamples.md). \
Once a PR is "ready to review", please trigger the workflows on 
[GitHub](https://github.com/owkin/MAKit/actions) 
to ensure these additional tests have succeeded. 

### Local 

Testing the [unit tests](../Architecture/MAKitTests.md) 
on Xcode is straight forward.

Testing the [integration tests](../Architecture/MATorchTests.md) 
and the [examples](../Architecture/MAExamples.md) requires an additional setup. 
Follow the previous links to know more about.

## Git Workflow

- We use the Github Flow. 
  Tags are used to indicate which commit of the Main branch 
  can be used in production.

- The Main branch is always protected so as the Release branches. 
  All updates to these branches arrive via pull requests (PRs).

- The name of the Main branch is `main`.

- The names of the Release branches are `release_N`, with N the number of the 
  current release. 
  The last Release branch contains the code under development.

- Each PR :

  - Must be reviewed by at least one maintainer of the repo, 
    implemented as code owners in GitHub (see [below](#code-owner)).

  - Should be as small and focused as possible.

  - Should implement changes that are backward compatible as much as possible.

  - Should provide the context of the changes and document what it solves.

- Push often changes to Github; at least on a daily basis 
  even if the PR is not yet ready for accepting reviews. 
  In this case, use the DRAFT status.

- TODOs can be left in the code if other PR deal with them. 
  In that case, add the link to those PR in the description. 
  If no PR deal with them, 
  it is better to use Github issues for small changes / improvements.

- In case of a new feature or bug fix, 
  unit tests should be added to cover all the use cases. 
  They should help reviewers understanding the PR.
  If such tests already exist, they should be mentioned in the description.

- Each important PR should update the CHANGELOG 
  (format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)) 
  or any other documentation (docstring, contributing, readme, ...).

- Squash before merge.

- Rebase your PR as often as possible, 
  if using `push force` then do it with lease and be careful about it 
  if several people work on the PR.

## Repo Admins, Code Owners, Authors

The key principle to grant roles to users is:
making sure workflow is smooth while limiting the number of people 
with extended rights for security purposes.

We detail below the responsibilities of each role and the processes to update 
the list of people with these roles.

### Admin

**Person** who has [admin right](
https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization) 
for a repository on GitHub.
"She/he has full access to the project, including sensitive and 
destructive actions like managing security or deleting a repository",
this is why it needs to be limited to a maximum of 3 people. 

### Code Owner

**Person** who knows well the code base and who is able to judge whether 
this can be merged in `main` or not.
The code owners have the following responsibilities:

- review PR before they can be merged into the `main` branch.
- handle GitHub Issues lifecycle.

Code owners are added to 
[this file](../../CODEOWNERS) by existing code owners, 
everybody can suggest new code owners. 

## Author

Person or organization that has written and merged at least one PR into MAKit.
Authors are added to [this file](../../AUTHORS).

Any author "MyName" is entitled to append the header of files where she/he had 
an impact: 

```swift
// 
// File.swift
// MAKit
//
// Created by ... on 28/10/2022.
// Modified by MyName on 28/10/2022.
//
```

Avoid to append your change if your name is already the latest appended and 
the last modification date is recent enough (< 1 month from now).

## Commit Message

The commit message must follow the 
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0) 
specification, so as the following 
[conventional emoji commits](
https://gist.github.com/parmentf/359667bf23e08a1bd8241fbf47ecdef0).

On top of the specification, we apply the following rules:

- recommended prefixes list: `fix`, `feat`, `build`, `chore`, `ci`, `docs`, 
  `style`, `refactor`, `perf` and `test`
- header line must be lower than 100 characters
- for breaking change, the commit must include 
  a footer with `BREAKING CHANGE:` and `!` after the type/scope

<ins>Examples</ins>: 

### With a Body

```
✨ feat: add options for auto batching in the CLI

Lorem ipsum dolor sit amet. Est soluta dolores rem itaque suscipit qui 
soluta porro 33 galisum rerum aut numquam voluptates qui Quis deserunt. 
```

### With a Scope

```
🐛 fix(core): add unique constraint on tag name
```

### With both ! and BREAKING CHANGE Footer

```
🔨 refactor!: remove GPU variable

BREAKING CHANGE: deprecate GPU variable
```

## Versioning

- MAKit adheres to [Semantic Versioning](https://semver.org).
- To bump the release version, follow the rules below:
  - X is at zero (MAKit is in development phase)
  - Y is bumped for new features, architecture changes 
    and non backwards compatible fixes
  - Z is bumped when backwards compatible bug fixes are introduced

## Release on GitHub

Let say we want to make the release X.Y.Z for the branch `release_N` 
containing the commits to merge into the `main` branch.

1. Ensure all [end-to-end tests](#ci) pass on `release_N`.

1. Update the Changelog
    - Create a branch from `release_N`.
    - Edit the changelog to create a new section 
      corresponding to the new release. 
      Move all the "Unreleased" items to this new section.
      Do not delete the "Unreleased" section title: future PRs will insert 
      changelog items in this section.
    - Commit and push the changes.
    - Squash and merge the new branch into `release_N` with commit message \
      🔧 chore: release X.Y.Z
      
1. Create a Pull Request

1. Review and Merge the Pull Request

1. Create a GitHub release X.Y.Z from `main`: 
     - GitHub > Releases > Draft new Release
       - **Choose a tag**: X.Y.Z (this will create a tag)
       - **Target**: main
       - **Release title**: X.Y.Z
       - **Describe this release**: formatted copy of the changelog 
         using the template provided in the 
         [changelog template](#changelog-template)

## CHANGELOG template

```md
## 0.Y.Z (2022-10-28)

### Features

### Bug Fixes

### Documentation

### Mischellaneous Tasks
```
