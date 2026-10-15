"""pytest 的项目根锚点。

它存在的唯一理由：这个文件所在的目录会被 pytest 插进 sys.path，
于是 `import src.xxx` 在测试里能用 —— 否则测试得靠改 sys.path 或装成包。
请勿删。
"""
