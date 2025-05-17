

# Creating conda envs

For HPC environments that have CUDA available as an `EasyBuild` module, you may need to first load CUDA:

```
module load EBModules CUDA/11.7.0
```

Then create the env:

`conda env create -f conda/<file.yml>`


# Installation GenVarLoader

## Install [`Pixi`](https://pixi.sh/latest/)
```
curl -fsSL https://pixi.sh/install.sh | sh
```

Clone GVL and run `Pixi`
```
git clone https://github.com/mcvickerlab/GenVarLoader.git

cd GenVarLoader

pixi run -e dev pre-commit
```

Activate Pixi
```
pixi shell
```
