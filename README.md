# How to Set Up MonsterVision2 for Development
This document covers installing MV2 on a Raspberry Pi development machine. See (???)) for deployment instructions.

It is recommeneded (but not required) that you use an SSD rather than an SD card on your Pi.  If you do, you may need to enable your Pi to boot from the SSD.  This only needs to be done once.  [Follow these instructions.](https://peyanski.com/how-to-boot-raspberry-pi-4-from-ssd/#:~:text=To%20boot%20Raspberry%20Pi%204%20from%20SSD%20you,USB%20to%20boot%20raspberry%20Pi%204%20from%20SSD.)

Once you've gotten your Pi up and running, follow this procedure:

## Start a Terminal session.
Within the session:

Make sure `pip` is up-to-date:
```shell
pip install --upgrade pip
``` 
Clone the MonsterVision2 repo:
```shell
git clone https://github.com/Lakemonsters2635/MonsterVision2.git
```
Change to the MonsterVision2 directory:
```shell
cd MonsterVision2
```

For development, it is best to use a Python virtual environment to keep from descending into "version hell."  Create the virtual environment and activate it.
```shell
virtualenv env
. env/bin/activate
```

Install all of the requirements for running:
```shell
pip install -r requirements.txt
```
Note that on a fresh system, the installation of OpenCV (opencv-contrib-python==4.5.5.62) may take several hours to build on an SSD-based system - even longer if you are using an SD card.

Finally, you'll need to copy 2 files into the `/boot` directory.  You'll need root permission to do this.  The first file is `frc.json` and contains:
```json
{
    "cameras": [],
    "ntmode": "client",
    "switched cameras": [],
    "team": 2635,
    "hasDisplay": 1
}
```
|Entry|Values||
|---|---|---|
|`ntmode`|**client**|Network tables server hosted remotely|
||**server**|Network tables server hosted locally|
|`team`|Team number||
|`hasDisplay`|**0**|Host is headless|
||**1**|Host has attached display - depth and annotation windows will be displayed|

Copy this file:
```shell
sudo cp frc.json /boot
```
The second needed in /boot is `nn.json`.  This file determines which detection network is to be run.  Copy one of the `.json` files from `./models` directory.  For example, to use the 2022 Cargo YOLOv6 Tiny network:
```shell
sudo cp models/nn-2022cargoyolo6t.json /boot/nn.json
```
Run MonsterVision2 via:
```shell
python MonsterVision2.py
```
## Development Environment
Visual Studio Code is the preferred development environment for consistency with our general FRC code development.  It is officially distributed via the Raspberry Pi OS APT repository in both 32- and 64-bit versions:  Install via:
```shell
sudo apt update
sudo apt install code
```
It can be launched with from a terminal via `code` or via the GUI under the **Programming** menu.
### Install ***GitHub Pull Requests and Issues*** Externsion
To enable GitHub integration, you need to install the above-named extension.  After installation, you'll need to log into your GitHub account.  **TODO** Instructions of doing this.
### Debugging Using VS Code
- After launching VS Code, select **File** | **Open Folder...** and select the `MonsterVision2` directory.
- Select **View** | **Command Palette...**.
- Choose `Python: Select Interpreter`.
- From the list of interpreters, choose the one in your virtual environment.  It will look something like this: `Python n.n.n ('env':venv) ./env/bin/python`
- From the Left pane, select `MonsterVision2.py` to open it.
Either hit `F5` or select **Run**|**Start Debugging** to run MonsterVision2 under control of the debugger.
