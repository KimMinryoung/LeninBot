"""/help and the Telegram "/" menu come from one command table."""
import unittest

from telegram import commands


class BotCommandTableTests(unittest.TestCase):
    def test_every_menu_command_is_documented_in_help(self):
        menu = commands.bot_menu_commands()
        self.assertEqual(len(menu), len({cmd for cmd, _ in menu}))
        for cmd, description in menu:
            with self.subTest(cmd=cmd):
                self.assertTrue(description)
                self.assertIn(f"/{cmd}", commands._HELP_TEXT)

    def test_menu_order_covers_every_menu_row(self):
        rows = {cmd for _sec, cmd, desc, _lines in commands._COMMANDS if desc}
        self.assertEqual(rows, set(commands.BOT_MENU_ORDER))


if __name__ == "__main__":
    unittest.main()
