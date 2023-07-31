# animatediff
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neggles/animatediff-cli/main.svg)](https://results.pre-commit.ci/latest/github/neggles/animatediff-cli/main)

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
# There's a nice pretty help screen with a bunch of info that'll print here.
```

If you can't work it out yourself from there, either I screwed something up, or this isn't
debugged and functional enough for you to use (yet).

## RiFE!

I have added experimental support for [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
using the `animatediff rife interpolate` command. It has fairly self-explanatory help, and it has
been tested on Linux, but I've **no idea** if it'll work on Windows.

Either way, you'll need ffmpeg installed on your system and present in PATH, and you'll need to
download the rife-ncnn-vulkan release for your OS of choice from the GitHub repo (above). Unzip it, and
place the extracted folder at `data/rife/`. You should have a `data/rife/rife-ncnn-vulkan` executable, or `data\rife\rife-ncnn-vulkan.exe` on Windows.

You'll also need to reinstall the repo/package with:
```py
python -m pip install -e '.[rife]'
```
or just install `ffmpeg-python` manually yourself.

Default is to multiply each frame by 8, turning an 8fps animation into a 64fps one, then encode
that to a 60fps WebM. (If you pick GIF mode, it'll be 50fps, because GIFs are cursed and encode
frame durations as 1/100ths of a second).

Seems to work pretty well...

## Credits:

see [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (very little of this is my work)

n.b. the copyright notice in `COPYING` is missing the original authors' names, solely because
the original repo (as of this writing) has no name attached to the license. I have, however,
used the same license they did (Apache 2.0).
