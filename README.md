# animatediff

animatediff refactor, because I can.

# LoRA loading is ABSOLUTELY NOT IMPLEMENTED YET!

## Also this currently only works with xformers and on a GPU.

I did what I could, okay? there's a lot of changes between Diffusers 1.11.1 and 1.18.0...

### How To Use

1. Lie down
2. Try not to cry
3. Cry a lot

### but for real?

Okay fine.

```sh
git clone https://github.com/neggles/animatediff-cli
cd animatediff-cli
python3.10 -m venv .venv
source .venv/bin/activate
# install Torch. Use whatever your favourite torch version >= 2.0.0 is, but, good luck on non-nVidia...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
# install the rest of all the things (probably! I may have missed some deps.)
python -m pip install -e '.[dev]'
# you should now be able to
animatediff --help
# and Typer will help u out. Kind of. A little.
```

If you can't work it out yourself from there, either I f$#@ed something up, or this isn't 
debugged and functional enough for you to use (yet).


## Credits:

see [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (very little of this is my work)

n.b. the copyright notice in `COPYING` is missing the original authors' names, solely because
the original repo (as of this writing) has no name attached to the license. I have, however,
used the same license they did (Apache 2.0).
