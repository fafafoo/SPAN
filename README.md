# SPAN
### SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction

#### SPAN
##### ├── create_folds.py
##### ├── Main_SPAN.py
##### ├── SPANModel.py
##### ├── Retest_SPAN.py
##### ├── RawDataProcess.py
##### ├── GraphDataProcess.py
##### ├── SysConfigruration.py
##### ├── logger_utils.py
##### ├── datasets
##### ├── logs
##### ├── models
##### ├── analysis
##### │   ├── plot_performance_comparison
##### │   ├── plot_case_studies
##### ├── python_env_pip.txt    (安装python虚拟环境 install python environment)
##### ├── requirements.txt
##### ├── SPAN.md
##### ├── LICENSE
##### └── batchretest.bat   (重现 retest)

##### 创建折叠数据集和分割文件, k 表示折叠的数量, round表示当前的轮次。

##### Create folds and split files, k indicates the number of folds, round represents the current round.

```Shell
python .\create_folds.py --k 5 --round 0         #(生成5折交叉验证的第0轮分割，并保存为文件。)
```

##### 运行模型训练和测试, k 表示折叠的数量, index 表示当前的折叠索引, round表示当前的轮次。

##### Run model training and testing, k indicates the number of folds, index indicates the current fold index, round represents the current round.

```Shell
python .\Main_SPAN.py --k 5 --index 0 --round 0          #(使用第0轮数据分割的第1个折叠进行训练和测试。)
```

##### 重测模型性能, k 表示折叠的数量, index 表示当前的折叠索引, round表示当前的轮次。

##### Retest model performance, k indicates the number of folds, index indicates the current fold index, round represents the current round.

```Shell
python .\Retest_SPAN.py --k 5 --index 0 --round 0
```

##### Example:

```Shell
python .\create_folds.py --k 5 --round 0
python -u .\Main_SPAN.py --k 5 --index 0 --round 0
python -u .\Retest_SPAN.py --k 5 --index 0 --round 0
```

##### my train commands for 5-fold cross validation of round 0~9:

```Shell
python -u ".\Main_SPAN.py" --k 5 --index 0 --round 0 > ./logs/stdout00.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 0 > ./logs/stdout01.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 0 > ./logs/stdout02.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 0 > ./logs/stdout03.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 0 > ./logs/stdout04.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 1 > ./logs/stdout10.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 1 > ./logs/stdout11.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 1 > ./logs/stdout12.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 1 > ./logs/stdout13.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 1 > ./logs/stdout14.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 2 > ./logs/stdout20.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 2 > ./logs/stdout21.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 2 > ./logs/stdout22.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 2 > ./logs/stdout23.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 2 > ./logs/stdout24.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 3 > ./logs/stdout30.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 3 > ./logs/stdout31.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 3 > ./logs/stdout32.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 3 > ./logs/stdout33.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 3 > ./logs/stdout34.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 4 > ./logs/stdout40.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 4 > ./logs/stdout41.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 4 > ./logs/stdout42.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 4 > ./logs/stdout43.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 4 > ./logs/stdout44.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 5 > ./logs/stdout50.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 5 > ./logs/stdout51.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 5 > ./logs/stdout52.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 5 > ./logs/stdout53.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 5 > ./logs/stdout54.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 6 > ./logs/stdout60.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 6 > ./logs/stdout61.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 6 > ./logs/stdout62.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 6 > ./logs/stdout63.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 6 > ./logs/stdout64.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 7 > ./logs/stdout70.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 7 > ./logs/stdout71.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 7 > ./logs/stdout72.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 7 > ./logs/stdout73.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 7 > ./logs/stdout74.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 8 > ./logs/stdout80.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 8 > ./logs/stdout81.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 8 > ./logs/stdout82.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 8 > ./logs/stdout83.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 8 > ./logs/stdout84.txt

python -u ".\Main_SPAN.py" --k 5 --index 0 --round 9 > ./logs/stdout90.txt
python -u ".\Main_SPAN.py" --k 5 --index 1 --round 9 > ./logs/stdout91.txt
python -u ".\Main_SPAN.py" --k 5 --index 2 --round 9 > ./logs/stdout92.txt
python -u ".\Main_SPAN.py" --k 5 --index 3 --round 9 > ./logs/stdout93.txt
python -u ".\Main_SPAN.py" --k 5 --index 4 --round 9 > ./logs/stdout94.txt
```

##### my retest commands for 5-fold cross validation of round 0~...... :

```Shell
python -u ".\Retest_SPAN.py" --k 5 --index 0 --round 0
python -u ".\Retest_SPAN.py" --k 5 --index 1 --round 0
python -u ".\Retest_SPAN.py" --k 5 --index 2 --round 0
python -u ".\Retest_SPAN.py" --k 5 --index 3 --round 0
python -u ".\Retest_SPAN.py" --k 5 --index 4 --round 0

python -u ".\Retest_SPAN.py" --k 5 --index 0 --round 1
......
```

##### 1. 重现实验结果，使用训练好的模型进行测试。

##### 1. Reproduce experimental results by testing the trained model.

```Shell
.\batchretest.bat #(retested results saved in ./logs as retest-round_x-fold_y.log)
```

##### 2. 汇总测试结果，生成统计表和性能比较图。请执行批处理程序或依次执行python脚本。

##### 2. Summarize test results, generate statistical tables and performance comparison plots. Execute batch processing programs or run python scripts sequentially.

```Shell
analysis\plot_performance_comparison\run_plot_performance_comparison.bat
# (result files saved in analysis\plot_performance_comparison\ as figure2.png and figure2.pdf)
```

##### 3. 进行案例分析，生成结果图和靶标建议报告。请先解压plot_case_studies.zip到analysis\plot_case_studies目录，然后执行批处理程序或依次执行python脚本。

##### 3. Conduct case analysis, generate result figure and target recommendation reports. Unzip plot_case_studies.zip to analysis\plot_case_studies directory first, then execute batch processing programs or run python scripts sequentially.

```Shell
analysis\plot_case_studies\run_plot_case_studies.bat
# (result files saved in analysis\plot_case_analysis\ as figure6.png and target_report_summary.txt)
```

---

#### License

SPAN is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Copyright (c) 2025 SPAN Authors (**Jian Liu, Chunxia Yin, Boyu Liu**)

You are free to use, modify, and distribute this software for **non-commercial purposes**,
including academic research and education, provided that appropriate credit is given
to the original authors.

**For commercial use, please contact the authors to obtain a commercial license.**

See [LICENSE](LICENSE) for the full license text, or visit
https://creativecommons.org/licenses/by-nc/4.0/
