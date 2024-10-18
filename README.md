# Calibration
一键标定内外参代码，含测距部分。
## Quick Start
```
git clone https://github.com/chakloong883/calibration.git
cd calibration
python3 calibration.py
```
内外参输出到param.yaml，测距验证图像输出到validate.png，每次重新标定内参需要手动删除param.yaml。
![image](validate.png)