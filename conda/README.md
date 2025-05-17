# Creating conda envs

For HPC environments that have CUDA available as an `EasyBuild` module, you may need to first load CUDA:

```
module load EBModules CUDA/11.7.0
```

Then create the env:

`conda env create -f conda/<file.yml>`

