import logging
import unittest

from common import subtitle, edl_util, constants


class SubtitleOperationsTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.srt_data = subtitle.read_subtitle_data(None, '../fixtures/audio_to_text.srt.txt')
        self.ass_data = subtitle.read_subtitle_data(None, '../fixtures/audio_to_text.ass.txt')
        self.maxDiff = None

    def test_cut_start_srt(self):
        subtitle.subtitle_cut(self.srt_data, 0, edl_util.parse_edl_ts('00:00:03'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:01,137 --> 00:00:04,596
the

2
00:00:08,211 --> 00:00:10,578
The lazy brown fox the

3
00:00:10,680 --> 00:00:13,308
The lazy brown fox the the

4
00:00:27,163 --> 00:00:29,655
The the
""")

    def test_cut_middle_srt(self):
        subtitle.subtitle_cut(self.srt_data, edl_util.parse_edl_ts('00:00:12'), edl_util.parse_edl_ts('00:00:13'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:00,500 --> 00:00:04,027
Hey there

2
00:00:04,137 --> 00:00:07,596
the

3
00:00:12,680 --> 00:00:15,308
The lazy brown fox the the

4
00:00:29,163 --> 00:00:31,655
The the
""")

    def test_cut_multiple_srt(self):
        subtitle.subtitle_cut(self.srt_data, edl_util.parse_edl_ts('00:00:12'), edl_util.parse_edl_ts('00:00:15'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:00,500 --> 00:00:04,027
Hey there

2
00:00:04,137 --> 00:00:07,596
the

3
00:00:27,163 --> 00:00:29,655
The the
""")

    def test_cut_end_srt(self):
        subtitle.subtitle_cut(self.srt_data, edl_util.parse_edl_ts('00:00:31'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:00,500 --> 00:00:04,027
Hey there

2
00:00:04,137 --> 00:00:07,596
the

3
00:00:11,211 --> 00:00:13,578
The lazy brown fox the

4
00:00:13,680 --> 00:00:16,308
The lazy brown fox the the
""")

    def test_cut_before_start_srt(self):
        subtitle.subtitle_cut(self.srt_data, 0, edl_util.parse_edl_ts('00:00:00.250'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:00,250 --> 00:00:03,777
Hey there

2
00:00:03,887 --> 00:00:07,346
the

3
00:00:10,961 --> 00:00:13,328
The lazy brown fox the

4
00:00:13,430 --> 00:00:16,058
The lazy brown fox the the

5
00:00:29,913 --> 00:00:32,405
The the
""")

    def test_cut_after_end_srt(self):
        subtitle.subtitle_cut(self.srt_data, edl_util.parse_edl_ts('00:00:40'), edl_util.parse_edl_ts('00:00:42'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_SRT, None, self.srt_data), """1
00:00:00,500 --> 00:00:04,027
Hey there

2
00:00:04,137 --> 00:00:07,596
the

3
00:00:11,211 --> 00:00:13,578
The lazy brown fox the

4
00:00:13,680 --> 00:00:16,308
The lazy brown fox the the

5
00:00:30,163 --> 00:00:32,655
The the
""")

    def test_cut_start_ssa(self):
        subtitle.subtitle_cut(self.ass_data, 0, edl_util.parse_edl_ts('00:00:03'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:01.14,0:00:04.60,Default,,0,0,0,,{TIME:1140,4600}the
Dialogue: 0,0:00:08.21,0:00:10.58,Default,,0,0,0,,{TIME:8210,10580}The lazy brown fox the
Dialogue: 0,0:00:10.68,0:00:13.31,Default,,0,0,0,,{TIME:10680,13310}The lazy brown fox the the
Dialogue: 0,0:00:27.16,0:00:29.65,Default,,0,0,0,,{TIME:27160,29650}The the
""")

    def test_cut_middle_ssa(self):
        subtitle.subtitle_cut(self.ass_data, edl_util.parse_edl_ts('00:00:12'), edl_util.parse_edl_ts('00:00:13'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.50,0:00:04.03,Default,,0,0,0,,{TIME:500,4030}Hey there
Dialogue: 0,0:00:04.14,0:00:07.60,Default,,0,0,0,,{TIME:4140,7600}the
Dialogue: 0,0:00:12.68,0:00:15.31,Default,,0,0,0,,{TIME:12680,15310}The lazy brown fox the the
Dialogue: 0,0:00:29.16,0:00:31.65,Default,,0,0,0,,{TIME:29160,31650}The the
""")

    def test_cut_multiple_ssa(self):
        subtitle.subtitle_cut(self.ass_data, edl_util.parse_edl_ts('00:00:12'), edl_util.parse_edl_ts('00:00:15'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.50,0:00:04.03,Default,,0,0,0,,{TIME:500,4030}Hey there
Dialogue: 0,0:00:04.14,0:00:07.60,Default,,0,0,0,,{TIME:4140,7600}the
Dialogue: 0,0:00:27.16,0:00:29.65,Default,,0,0,0,,{TIME:27160,29650}The the
""")

    def test_cut_end_ssa(self):
        subtitle.subtitle_cut(self.ass_data, edl_util.parse_edl_ts('00:00:31'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.50,0:00:04.03,Default,,0,0,0,,{TIME:500,4030}Hey there
Dialogue: 0,0:00:04.14,0:00:07.60,Default,,0,0,0,,{TIME:4140,7600}the
Dialogue: 0,0:00:11.21,0:00:13.58,Default,,0,0,0,,{TIME:11210,13580}The lazy brown fox the
Dialogue: 0,0:00:13.68,0:00:16.31,Default,,0,0,0,,{TIME:13680,16310}The lazy brown fox the the
""")

    def test_cut_before_start_ssa(self):
        subtitle.subtitle_cut(self.ass_data, 0, edl_util.parse_edl_ts('00:00:00.250'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.25,0:00:03.78,Default,,0,0,0,,{TIME:250,3780}Hey there
Dialogue: 0,0:00:03.89,0:00:07.35,Default,,0,0,0,,{TIME:3890,7350}the
Dialogue: 0,0:00:10.96,0:00:13.33,Default,,0,0,0,,{TIME:10960,13330}The lazy brown fox the
Dialogue: 0,0:00:13.43,0:00:16.06,Default,,0,0,0,,{TIME:13430,16060}The lazy brown fox the the
Dialogue: 0,0:00:29.91,0:00:32.40,Default,,0,0,0,,{TIME:29910,32400}The the
""")

    def test_cut_after_end_ssa(self):
        subtitle.subtitle_cut(self.ass_data, edl_util.parse_edl_ts('00:00:40'), edl_util.parse_edl_ts('00:00:42'))
        self.assertEqual(subtitle.write_subtitle_data(constants.CODEC_SUBTITLE_ASS, None, self.ass_data), """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.50,0:00:04.03,Default,,0,0,0,,{TIME:500,4030}Hey there
Dialogue: 0,0:00:04.14,0:00:07.60,Default,,0,0,0,,{TIME:4140,7600}the
Dialogue: 0,0:00:11.21,0:00:13.58,Default,,0,0,0,,{TIME:11210,13580}The lazy brown fox the
Dialogue: 0,0:00:13.68,0:00:16.31,Default,,0,0,0,,{TIME:13680,16310}The lazy brown fox the the
Dialogue: 0,0:00:30.16,0:00:32.65,Default,,0,0,0,,{TIME:30160,32650}The the
""")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
