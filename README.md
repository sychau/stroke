# SFU CMPT 340 Project -- AmIHavingAStroke.ca
A common symptom of strokes is facial drooping and asym-
metry. We present a Python program that aims to help patients assess
their own health by analysing their face and determining the likelihood
of a positive diagnosis. We pre-process and detect facial landmarks of the
dataset, consisting of stroke patients and healthy individuals, with the
dlib library. We then extract features from the landmarks, and finally we
train an SVM model and use it to predict if the user is experiencing a
stroke. The experimental results of our system average to 93%, indicat-
ing a high accuracy in facial droop recognition, which help identifying
patients having a stroke.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/kabhishe_sfu_ca/EbkOTT_AH9hIsy5At01xZN0BVhN6bbBzvbeen9o5dtNhhQ?e=qZ4fcS) | [Slack channel](https://app.slack.com/client/T05JYJAF22G/C05TT16EL6M/docs/Qp:F05T7QMRAVC/1701376363816) | [Project report](https://www.overleaf.com/project/650ca3366716f07f3579d8ce) |
|-----------|---------------|-------------------------|


## Video/demo/GIF

https://github.com/sfu-cmpt340/project_21/assets/64757355/dfbcf596-00b1-4d6c-ae25-d8c468343eb8

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code stroke detection program
├── dataset                      ## dataset that train and test the model
├── model                        ## pre-traind machine learning models
├── scripts                      ## scripts, train and save the model to model directory
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate $THISREPO
```

<a name="repro"></a>
## 3. Reproduction
To train the model
```bash
cd project_21
python scripts/train_model_svm.py
```
This will save the model to model directory. The model evaluation result will be printed in the terminal.

To run the GUI
```bash
cd project_21
python src/UI/UI.py
```

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
