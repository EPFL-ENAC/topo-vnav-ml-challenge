"""
This file is based on the codalab template
---
regis.longchamp@epfl.ch
"""
import json
import sys
import os.path
from evaluation import scoring


input_dir = sys.argv[1]
output_dir = sys.argv[2]
submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')


if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create result and html file
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    output_filename_html = os.path.join(output_dir, 'scores.html')
    output_file_html = open(output_filename_html, 'w')

    # run results evaluation
    ai_competition_gt_file_path = os.path.join(truth_dir, "gt.csv")
    ai_competition_result_file_path = os.path.join(submit_dir, "est.csv")
    score = scoring(ai_competition_result_file_path,ai_competition_gt_file_path)

    # write evaluation
    if score.get('formatting').get('succeed') :
        output_file.write("correct:%s"%score.get('score_overall'))
    else :
        output_file.write("correct:0")
    html = json.dumps(score)
    output_file_html.write(html)
    output_file_html.close()
    output_file.close()