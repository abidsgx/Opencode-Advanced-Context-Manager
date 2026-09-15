This plugin is designed to manage context windows for agents, and improve responses. It also stores session data for your agents to reference in the future.

## Main Features

### Machine learning 

A machine learning model (SGDClassifier) is used to classify which files and components of the code are important for the current goal of the project, based on features such time spent on editing, lines modified, and number of dependent files. All features can be found in /server/features/base.py.

### Active learning

Active learning can improve the models predictions by manually classifying uncertain predictions. You can enable this feature in /data/config.yaml by setting `online learning: enabled: true`

After each session ends, you can type /ctx feedback to classify uncertain predictions from 0-10.

### Importance based pruning

A priority map is built from the importance scores. Messages referencing low importance files/components of your code will be removed from the context window or compressed. 

Pruning configured in /plugin/lib/context/prune.ts.

### Session recording and data storage

Prompts, agent responses, edits, relevant files, and the relationships between files/ functions are stored for future reference if the agents forget key information. 

Session recorder is in plugin/lib/session/recorder.ts.

### Slash commands

/ctx help — Show available commands
/ctx stats — Token usage, pruning stats, importance distribution
/ctx importance [file] — Show importance scores for a file/function
/ctx matrix [file] — Show connections to other files
/ctx feedback — Active learning: present uncertain predictions for labeling
/ctx session — Show current session data summary
/ctx config — Show/edit configuration

## Setup

First download and extract the zip file.

Go into the .opencode/opencode.json project configuration file, or the \.config\opencode\opencode.jsonc global configuration file and add the folder location to the plugin array:

```
{
  "plugin": [
    ...
    "folder path"
  ]
}

```




