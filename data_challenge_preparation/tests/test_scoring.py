import os,sys
import pytest


myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from evaluation import scoring


def test_1_ground_truth_vs_ground_truth_() :
    """
    Test the result if the ground truth is used alsa as estimation file
    """

    path_ground_truth = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','t1_gt.csv')
    path_est  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','t1_gt.csv')

    result = scoring(path_est,path_ground_truth)

    assert result.get('score_error') == 100
    assert result.get('score_overall') == 70
    assert result.get('median_error_on_coordinates') == 0
    assert result.get('median_error_on_angles') == 0


def test_2_sample_data() :
    """
    Test with the following parameters :
    - uncertainty = 1 for all 6d degree of freedom
    - same coordinate
    - one degree difference
    see https://docs.google.com/spreadsheets/d/1sYLlQ2cLDg3JtRNFkJ4pcZ9iKjvMGN2s/edit#gid=1181985230 for example
    """

    expected_values = dict()
    expected_values['median_error_on_coordinates'] = 0
    expected_values['median_error_on_angles'] = 1.5164113250328997 # can not be tested !
    expected_values['score_error'] = 96.96717735 # (100-0)*0.7 + (15-1.5164113250328997)/15*100*0.3
    expected_values['uncertainty_on_coordinates'] = 0 # validated by excel 
    expected_values['uncertainty_on_angles'] = 0.6561787149247866 # validated by excel
    expected_values['score_uncertainty'] = 30 # 0*0.7 + 1*0.3 (0.6561787149247866 in the range -> 1)
    expected_values['score_overall'] = 76.877 # 96.96717735*0.7 + 30*0.3



    path_ground_truth = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','t2_gt.csv')
    path_est  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','t2_est.csv')

    result = scoring(path_est,path_ground_truth)

    for key, expected_value in expected_values.items() :
        computed_value = result.get(key)
        assert round(abs(computed_value-expected_value),5) == 0

