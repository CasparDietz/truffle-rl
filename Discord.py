import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
plt.style.use('ggplot')
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from typing import Any
from discord_webhook import DiscordWebhook
from scipy.special import binom
from collections import defaultdict

def send_msg_to_discord_channel(content: str, url=str):
    print(content)
    try:
        if url != '':
            webhook = DiscordWebhook(
                url=url,
                content=content,
            )
            webhook.execute()
    except Exception as e:
        print(e)

def send_msg_with_file_to_discord_channel(content: str, file: Any, filename: str, url=str):
    try:
        if url != '':
            webhook = DiscordWebhook(
                url=url, content=content
            )
            webhook.add_file(file=file,
                            filename=filename)
            webhook.execute()
    except Exception as e:
        print(e)