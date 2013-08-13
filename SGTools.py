"""
Skullgirls Tools.

Usage:
    SGTools.py sprite indexed [options] [--no-lines]          <src> <msb> <palette> <output_folder>
    SGTools.py sprite         [options] [--channel=<channel>] <src> <msb> <output_folder>
    SGTools.py sprite info    <msb>
    SGTools.py -h | --help
    SGTools.py --version

Any source images must be in a common format (TGA, PNG, etc).

Arguments:
    <src>                       Sprite texture file.
    <msb>                       Sprite msb file.
    <output_folder>             Folder to dump frames in.

Common Options:
    -a --animation=<name>       Animation to dump, leave blank for all frames.
    --bg=<colour>               Background colour as 4-tuple [default: (255, 255, 255, 0)].
    --tile-bg=<tile_colour>     Tile background colour as 4-tuple, copies background by default.
    -h --help                   Show this screen.
    --version                   Show version.

Palette Options:
    --no-lines                  Disable line art when in palette mode.

Non-Palette Options
    --channel=<channel>         Only output specified channel (r, g, or b).

"""

__all__ = ["Sprite", "Frame", "Animation"]
__license__ = "MIT"
__version__ = "0.0.1"
__status__ = "Development"

from collections import namedtuple
import os.path
import math
import struct
import warnings

import numpy as np
from PIL import Image as PILImage
from PIL.ImageDraw import Draw as PILDraw


Animation = namedtuple('Animation', 'offset length')
Frame = namedtuple('Frame', 'offset length duration')


class Sprite(object):
    """This class extracts frames from sprite files with an msb file.

    Method Summary:
    get_processed_image  -- Used to get the intermidiary tile sheet.
    get_frame            -- Used to get a single frame of animation.
    dump_animation       -- Used to write all frames of animation to files.

    """
    name = None
    tile_w = None
    tile_h = None
    tiles = []
    frames = []
    animations = {}
    img = None
    _processed_img = None
    palette = None
    palette_width = None
    palette_height = None

    def __init__(self, msb_filename, img_filename=None,
                 palette_filename=None, limit_channel=None, lines=True,
                 bg=(255, 255, 255, 0), bg_lerp=None):
        """Initialize Sprite representation.

        Arguments:
        msb_filename     -- str, path to sprite msb.
        img_filename     -- str, path to texture.
        palette_filename -- str, path to palette.
                              Will be ignored if channel is limited.
        limit_channel    -- str, channel to output. Omit to composite all
                              channels. Will override palette settings.
        lines            -- bool, multiply lineart over output.
                              Will be ignored if palette is not used.
        bg               -- 4-tuple, background colour for areas without a tile.
        bg_lerp          -- 4-tuple, background colour for areas with a tile.
                              If omitted it will copy the value of bg.

        """
        # Open images
        if img_filename:
            self.img = PILImage.open(img_filename)
        if palette_filename:
            self.palette = PILImage.open(palette_filename)
            self.palette_width, self.palette_height = self.palette.size

        # Load msb
        self.data = self._load_msb(msb_filename)
        self._parse_msb()

        # Misc settings
        self._limit_channel = limit_channel
        self._draw_lines = lines

        # Backgrounds are separate to allow for easy display of tiles
        self.bg = bg
        self.bg_lerp = bg_lerp
        if self.bg_lerp is None:
            self.bg_lerp = self.bg
        # bg_lerp must have range [0, 1]
        self.bg_lerp = tuple(x / 255.0 for x in self.bg_lerp)

    # Helpers for reading data
    @staticmethod
    def _load_msb(filename):
        """Return filename as a list of bytes."""
        with open(filename, 'rb') as f:
            data = f.read()
        return [ord(x) for x in data]

    def _take(self, n):
        "Return n bytes read from the msb."
        val = self.data[:n]
        assert len(val) == n
        self.data = self.data[n:]
        return val

    def _get_long(self):
        "Return an 8-byte unsigned long long read from the msb."
        val = self._take(8)
        val = struct.unpack('>Q', struct.pack('8B', *val))
        return val[0]

    def _get_int(self):
        "Return a 4-byte unsigned int read from the msb."
        val = self._take(4)
        val = struct.unpack('>I', struct.pack('4B', *val))
        return val[0]

    def _get_string(self):
        "Return a Pascal-style string read from the msb."
        length = self._get_long()
        string = self._take(length)
        string = ''.join(chr(x) for x in string)
        return string

    def _parse_msb(self):
        """Parse all fields from the msb file."""

        self._take(11)  # Skip signature
        self.name = self._get_string()
        # Unknown
        self._get_int()
        self._get_string()
        self._get_long()
        # Sizes of next data sections
        tile_count = self._get_long()
        frame_count = self._get_long()
        animation_count = self._get_long()
        # Tile dimensions
        self.tile_w = self._get_long()
        self.tile_h = self._get_long()
        # Tiles
        tiles = []
        for _ in range(tile_count):
            tile = self._take(4)
            tiles.append(tuple(tile))
        # Frames
        frames = []
        for _ in range(frame_count):
            offset = self._get_int()
            length = self._get_int()
            duration = self._get_int()
            self._take(8)  # Unknown
            frames.append(Frame(offset, length, duration))
        # Animations
        animations = {}
        for _ in range(animation_count):
            name = self._get_string()
            start = self._get_int()
            length = self._get_int()
            self._get_int()  # TODO: Unknown
            last = self._get_int()
            calculated_length = last + 1 - start
            if length != calculated_length:
                warnings.warn("Length mismatch in animation definition '%s': %s vs %s"
                              % (name, length, calculated_length), RuntimeWarning)
            animations[name] = Animation(start, length)

        self.tiles = tiles
        self.frames = frames
        self.animations = animations

    @staticmethod
    def _mix_layers(src, dest, output):
        """Composite src layer onto dest layer, store result in output."""

        def get_channel(layer, channel):
            """Helper, retrieve channel from the 4-tuple or numpy array."""
            if isinstance(layer, (tuple, list)):
                return layer[channel]
            else:
                return layer[:,:,channel]

        a1, a2 = get_channel(dest, 3), get_channel(src, 3)

        # Mix colour channels
        for channel in range(3):
            output[:,:,channel] = (get_channel(src, channel) * a2
                                   + get_channel(dest, channel) * a1 * (1 - a2))

        # Correct alpha
        output[:,:,3] = a2 + a1 * (1 - a2)
        non_zero = np.nonzero(output[:,:,3])
        for channel in range(3):
            output[:,:,channel][non_zero] /= output[:,:,3][non_zero]

    def get_processed_image(self):
        """Apply processing steps to tile sheet.

        Processing is dependent on the constructor arguments given.
        If limit_channel was set, only one channel will be output.
        If a palette was set it will be used to generate a corrected image.
        """

        # Check cache
        if self._processed_img is not None:
            return self._processed_img

        # Prepare palette
        if self.palette is not None:
            palette = np.array(self.palette.convert('RGBA'))
        else:
            palette = None

        # Convert image to RGBA array
        if self.img is None:
            raise RuntimeError("No image loaded.")
        src = self.img.convert('RGBA')
        src = np.array(src)

        if self._limit_channel is not None:
            # Determine which channel to duplicate from
            channel = self._limit_channel
            channel = 0 if channel == 'r' else 1 if channel == 'g' else 2 if channel == 'b' else -1
            if channel == -1:
                raise RuntimeError("Invalid channel specified: %s" %
                                   self._limit_channel)
            src_channel = src[:,:,channel]

            # Determine which channels to duplicate to
            channels = set([0, 1, 2])
            channels -= set([channel])

            # Duplicate channel
            for other_channel in channels:
                src[:,:,other_channel] = src_channel

            # Force full alpha
            src[:,:,3] = 255
        elif palette is not None:
            # Line art processing
            lineart_channel = src[:,:,0].astype(float).copy() / 255.0

            # Get final colour using palette
            # Steps:
            # 1. Convert every pixel to a uv pair for the palette
            # 2. Replace all pixels with the uv-indexed palette colours
            # 3. Composite onto background
            # 4. Composite line art on top

            # Luminosity from blue channel
            # Scale to palette width
            uv_xs = src[:,:,2].clip(0, 254)
            uv_xs *= self.palette_width / 255.0
            uv_xs = uv_xs.flatten()
            # Palette index from green channel
            # Divide by 4
            #   TODO: This value isn't from the msb, it may not always work
            # Quick check to see if 4 will result in clipping
            p_max = np.amax(src[:,:,1])
            p_min = np.amin(src[:,:,1][np.nonzero(src[:,:,1])])
            palette_step = int(math.ceil(float(p_max) / (self.palette_height - 1)))
            if palette_step > 4:
                warnings.warn(
                    "Warning: Palette step is too small, palette clipping will occur."
                    + "\nGreen range: %s - %s" % (p_min, p_max)
                    + "\nCalculated palette step: %s" % palette_step
                    + "\nUsing palette step: 4",
                    RuntimeWarning
                )
            uv_ys = src[:,:,1]
            uv_ys /= 4
            uv_ys = uv_ys.clip(0, self.palette_height - 1)
            uv_ys = uv_ys.flatten()
            # Colours from uvs
            uvs = np.array([uv_ys, uv_xs]).T
            colours = palette[uvs[:,0], uvs[:,1]]
            colours.shape = src.shape
            src = colours

        # Final background colour pass and line art (if applicable)
        if self._limit_channel is None:
            # Convert channels to floats
            src = src.astype(float)
            src /= 255.0

            # Mix bg_lerp in
            self._mix_layers(src, self.bg_lerp, src)

            if palette is not None:
                # Multiply by lineart
                if self._draw_lines:
                    self._mix_layers((0, 0, 0, 1 - lineart_channel), src, src)

            # Convert channels to bytes
            src *= 255.0
            src = src.astype('uint8')

        src = PILImage.fromarray(src)

        # Cache
        self._processed_img = src
        return src

    def get_frame(self, n, im_width, im_height):
        """Return a single frame as a PIL Image.

        Arguments:
        n         -- the frame index
        im_width  -- the width of the Image to return
        im_height -- the height of the Image to return

        """
        offset, length, duration = self.frames[n]
        tiles = self.tiles[offset:offset + length]

        # Create new blank image
        img = PILImage.new('RGBA', (im_width, im_height))
        draw = PILDraw(img)
        draw.rectangle(((0, 0), (im_width, im_height)), fill=self.bg)

        src = self.get_processed_image()

        # Blit tiles
        for tile in tiles:
            x, y, u, v = tile
            # Fix edge repeat by overlapping tiles 1px
            x *= self.tile_w - 1
            y *= self.tile_h - 1
            if x > 0:
                x += 1
            if y > 0:
                y += 1
            # Crop tile
            u *= self.tile_w
            v *= self.tile_h
            box = (u, v, u + self.tile_w, v + self.tile_h)
            part = src.crop(box)
            # Paste onto frame
            box = (x, y, x + self.tile_w, y + self.tile_h)
            img.paste(part, box)
        return img

    def dump_animation(self, target_dir, animation_name=None):
        """Dump frames of animation as pngs.

        Frames will be saved with the pattern %03d.png.

        Arguments:
        target_dir     -- directory to save frames in
        animation_name -- animation to dump

        """
        tiles, frames = self.tiles, self.frames
        if animation_name is not None:
            anim = self.animations[animation_name]
        else:
            # All frames
            anim = Animation(0, len(frames))
            # animation_name = 'all'
        start, n = anim
        # Calculate required image size to hold all frames
        all_tiles = [tile for frame in frames[start:start + n] for tile in
                     tiles[frame.offset:frame.offset + frame.length]]
        max_x = max(tile[0] for tile in all_tiles)
        max_y = max(tile[1] for tile in all_tiles)
        im_width = (16 - 1) * (max_x + 1) + 1
        im_height = (16 - 1) * (max_y + 1) + 1

        # Dump images to target directory
        frame_info = []
        for i in range(n):
            out_img = self.get_frame(start + i, im_width, im_height)
            out_img.save(os.path.join(target_dir, '%03d.png' % i))
            frame_info.append(['%03d.png' % i, frames[i].duration])
        return frame_info


def _command_line():
    """Execute command-line interface."""
    from docopt import docopt, DocoptExit
    from ast import literal_eval

    def _parse_colour(colour_string):
        """Return colour_string as a 4-tuple colour if possible."""
        if colour_string is None:
            return colour_string
        try:
            tup = literal_eval(colour_string)
            assert isinstance(tup, tuple)
            assert len(tup) == 4
            assert all(0 <= x < 256 for x in tup)
            assert all(isinstance(x, int) for x in tup)
        except Exception:
            # Use docopt error reporting
            raise DocoptExit('Invalid colour: ' + colour_string)
        return tup

    args = docopt(__doc__, version="SGTools v%s" % __version__)

    if args['sprite']:
        if args['info']:
            # Print sprite info
            sprite = Sprite(msb_filename=args['<msb>'])
            print 'Sprite Name:     ', sprite.name
            print 'Tile Size:       ', sprite.tile_w, 'x', sprite.tile_h
            print 'Tile Definitions:', len(sprite.tiles)
            print 'Frames:          ', len(sprite.frames)
            print 'Animations:      ', len(sprite.animations)
            print 'Animation names: '
            print '\n'.join('    ' + name for name in sprite.animations)
        else:
            constructor_args = {
                'img_filename': args['<src>'],
                'msb_filename': args['<msb>'],
                'palette_filename': args['<palette>'],  # Possibly None
                'limit_channel': args['--channel'],  # Possibly None
                'lines': not args['--no-lines'],
                'bg_lerp': _parse_colour(args['--tile-bg']),
            }
            if args['--bg']:
                constructor_args['bg'] = _parse_colour(args['--bg'])

            sprite = Sprite(**constructor_args)
            sprite.dump_animation(args['<output_folder>'], args['--animation'])


if __name__ == '__main__':
    _command_line()
