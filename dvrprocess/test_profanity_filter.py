import re
import unittest
import profanity_filter


class ProfanityFilterTest(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.censor_list = profanity_filter.load_censor_list()
        self.stop_list = profanity_filter.load_stop_list()

    def test_filter_text(self):
        """
        Tests for a phrase that also contains a single word
        """
        text = 'he acts like a bad-ass most of the time'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('he acts like a *** most of the time', filtered)

    def test_filter_text_with_markup(self):
        """
        Tests for a phrase that also contains a single word in markup
        """
        text = '{\\an7}{\\pos(76,228)}it is a damn\\N{\\an7}{\\pos(153,243)}shame'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('{\\an7}{\\pos(76,228)}it is a ***\\N{\\an7}{\\pos(153,243)}shame', filtered)

    def test_filter_text2(self):
        """
        Tests for a single word censor
        """
        text = 'it is a damn shame'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('it is a *** shame', filtered)

    def test_filter_text3(self):
        """
        Tests for a hypenated word
        """
        text = 'the shit-storm will be terrible'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('the *** will be terrible', filtered)
        text = 'the shit storm will be terrible'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('the *** will be terrible', filtered)
        text = 'the shitstorm will be terrible'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('the *** will be terrible', filtered)

    def test_filter_text_with_markup2(self):
        """
        Tests for a single word censor in markup
        """
        text = '{\\an7}{\\pos(76,228)}what the\\N{\\an7}{\\pos(153,243)}hell is going on?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('{\\an7}{\\pos(76,228)}what ***\\N{\\an7}{\\pos(153,243)}*** is going on?', filtered)

    def test_stop(self):
        text = 'talking about boobs is a stop phrase!'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('***', filtered)

    def test_stop_with_markup(self):
        text = '{\\an7}{\\pos(76,228)}do not talk \\N{\\an7}{\\pos(153,243)}about boobs !'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('{\\an7}{\\pos(76,228)}***', filtered)

    def test_stop_no_subpattern(self):
        text = 'talking about cockatiels is ok !'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('talking about cockatiels is ok !', filtered)

    def test_exclamation(self):
        text = '{\\an7}{\\pos(76,228)} God, what is this?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('{\\an7}{\\pos(76,228)} *** what is this?', filtered)

    def test_exclamation2(self):
        text = '{\\an7}{\\pos(115,228)}\\h\\h\\h\\h\\h\\h\\h\\h{\\i1}\\hGod! you were--\\N{\\an7}{\\pos(115,243)}{\\i0}??{\\i1} We totally got you--'
        expected = '{\\an7}{\\pos(115,228)}\\h\\h\\h\\h\\h\\h\\h\\h{\\i1}\\h*** you were--\\N{\\an7}{\\pos(115,243)}{\\i0}??{\\i1} We totally got you--'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_quoted(self):
        text = '{\\an7}{\\pos(76,228)} "God, what is this?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual('{\\an7}{\\pos(76,228)} "*** what is this?', filtered)

    def test_exclamation_quoted2(self):
        text = r'{\an7}{\pos(38,28)}??"God, we???ve never worked\N{\an7}{\pos(38,44)}??in place like this before.'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(r'{\an7}{\pos(38,28)}??"*** we???ve never worked\N{\an7}{\pos(38,44)}??in place like this before.', filtered)

    def test_exclamation_phrase(self):
        text = r'{\an7}{\pos(115,243)}??Oh, my God.'
        expected = r'{\an7}{\pos(115,243)}??***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase2(self):
        text = r"{\\an7}{\\pos(86,243)}God..."
        expected = r"{\\an7}{\\pos(86,243)}***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase3(self):
        text = r'{\an7}{\pos(115,243)}God'
        expected = r'{\an7}{\pos(115,243)}***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase4(self):
        text = r'God'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase5(self):
        text = r'God!'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase6(self):
        text = r' God'
        expected = r' ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'  God'
        expected = r'  ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase7(self):
        text = r'son of a ...'
        expected = r'*** '
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'son of a bitch'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'son of a gun'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase8(self):
        text = r'{\an7}{\pos(115,243)}??My God, what happened?'
        expected = r'{\an7}{\pos(115,243)}??*** what happened?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase9(self):
        text = r'Yes! But, by God,'
        expected = r'Yes! But, ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase10(self):
        text = r'oh, god almighty, lady'
        expected = r'*** lady'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase11(self):
        text = r'god almighty!'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase12(self):
        text = r'oh, crap!'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase_wildcards1(self):
        text = r'What the f'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What the f..'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What the fu..'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What the fu..!'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_phrase_wildcards2(self):
        text = r'you screwed the pooch'
        expected = r'you ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'screw the pooch'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase1(self):
        text = r'God bless you.'
        expected = r'God bless you.'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase2(self):
        text = r'God\'s plan?'
        expected = r'God\'s plan?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase3(self):
        text = r"{\\an7}{\\pos(38,228)}and pray for God???s\\N{\\an7}{\\pos(38,243)}protection."
        expected = r"{\\an7}{\\pos(38,228)}and pray for God???s\\N{\\an7}{\\pos(38,243)}protection."
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase4(self):
        text = r"{\\an7}{\\pos(86,228)}Oh, thank God,\\N{\\an7}{\\pos(86,243)}I can???t wait."
        expected = r"{\\an7}{\\pos(86,228)}Oh, thank God,\\N{\\an7}{\\pos(86,243)}I can???t wait."
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase5(self):
        text = r"Lord Tennison"
        expected = r"Lord Tennison"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase6(self):
        text = r"{\\an7}{\\pos(134,228)}\\h\\h\\h\\his not praying\\N{\\an7}{\\pos(134,243)}to Ugbuh, the bug god."
        expected = r"{\\an7}{\\pos(134,228)}\\h\\h\\h\\his not praying\\N{\\an7}{\\pos(134,243)}to Ugbuh, the bug god."
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase7(self):
        text = r"son of a rich man"
        expected = r"son of a rich man"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase8(self):
        text = r"I am Kahmunrah. I am half god,"
        expected = r"I am Kahmunrah. I am half god,"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase9(self):
        text = r"{\\an7}{\\pos(134,228)}\\h\\h\\h\\his not praying\\N{\\an7}{\\pos(134,243)}to Ugbuh, the bug god."
        expected = r"{\\an7}{\\pos(134,228)}\\h\\h\\h\\his not praying\\N{\\an7}{\\pos(134,243)}to Ugbuh, the bug god."
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_exclamation_allowed_phrase10(self):
        text = r"may god almighty help us"
        expected = r"may god almighty help us"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_substring(self):
        text = r'In a nutshell, as you know,'
        expected = r'In a nutshell, as you know,'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_substring2(self):
        text = r'Using a hellfire missile'
        expected = r'Using a hellfire missile'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_pre_filtered(self):
        """
        Some broadcasts have the subtitles filtered but not the audio
        """
        text = r'What the - - - -'
        expected = r'What the ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What the !@#$%'
        expected = r'What the ***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What the ***'
        expected = r'What the *** _'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        text = r'What happened ?!?'
        expected = r'What happened ?!?'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_single_quote(self):
        text = r"for Christ's sake"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_dot(self):
        text = r"where's my freakin' gun"
        expected = r"where's my *** gun"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)

    def test_dot2(self):
        text = r"where's my freakin gun"
        expected = r"where's my *** gun"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertFalse(stopped)

    def test_hypen1(self):
        text = r'piss-off'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertFalse(stopped)
        text = r'piss off'
        expected = r'***'
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertFalse(stopped)

    def test_phrase_to_pattern_single_world(self):
        phrase = "^god$"
        pattern = profanity_filter.phrase_to_pattern(phrase)
        # print(pattern)
        self.assertFalse(re.search(pattern, 'half god,', re.IGNORECASE))

    def test_dot_splat(self):
        text = r"take your clothes off"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take all your clothes off"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take all your damn clothes off"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take off your clothes"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take off all your clothes"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take off your damn clothes"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take off his clothes"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)
        text = r"take off her clothes"
        expected = r"***"
        filtered, stopped = profanity_filter.filter_text(self.censor_list, self.stop_list, text)
        self.assertEqual(expected, filtered)
        self.assertTrue(stopped)


if __name__ == '__main__':
    unittest.main()
