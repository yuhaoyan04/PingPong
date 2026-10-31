import importlib.util
from pathlib import Path
import numpy as np

SPEC=importlib.util.spec_from_file_location('refine_observations_tool',Path(__file__).parents[1]/'tools/refine_observations.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def measurements():
    times=np.arange(17,dtype=np.int64)*500
    seed=np.tile([100.,100.,35.],(17,1));circle=np.array([101.,101.,36.])
    theta=(np.arange(72)+.5)*2*np.pi/72
    points=circle[:2]+circle[2]*np.c_[np.cos(theta),np.sin(theta)]
    records=[{'proposed_circle':circle.tolist(),'validation_after_inlier_fraction':1.,
              'validation_after_median_px':.1,'independent_arc_disagreement_px':.1,'condition':1.4,
              'edge_points':points.tolist(),'edge_sector_ids':list(range(72))} for _ in times]
    return times,seed,records


def test_short_geometry_gap_keeps_continuous_model_without_an_old_circle_jump():
    times,seed,records=measurements();records[8]={'proposed_circle':None}
    out,regime=M.continuous_circles(times,seed,records)
    np.testing.assert_allclose(out,np.tile([101.,101.,36.],(17,1)),atol=1e-9)
    assert len(set(regime))==1
    assert records[8]['continuous_status']=='short_gap_temporal_geometry'
    assert not records[8]['geometry_current_arcs_supported']


def test_strong_current_boundary_contradiction_excludes_spin_without_warp_jump():
    times,seed,records=measurements();records[8]['proposed_circle']=None
    points=np.array(records[8]['edge_points']);records[8]['edge_points']=(101+(points-101)*1.12).tolist()
    out,_=M.continuous_circles(times,seed,records)
    np.testing.assert_allclose(out[8],[101,101,36],atol=1e-9)
    assert records[8]['final_accepted']
    assert not records[8]['geometry_rotation_usable']


def test_insufficient_neighbor_measurements_never_create_a_new_geometry():
    times,seed,records=measurements()
    for i in range(len(records)):
        if i not in [3,8]:records[i]={'proposed_circle':None}
    out,_=M.continuous_circles(times,seed,records)
    np.testing.assert_array_equal(out,seed)
    assert not any(r['final_accepted'] for r in records)


def test_frame_time_coordinate_transform_preserves_sensor_position():
    old=np.array([700.,300.,40.]);new=np.array([702.,299.,42.]);scale=1.2
    row={'center_x_px':'160','center_y_px':'110','x1':'150','y1':'100','x2':'170','y2':'120','position_estimate_method':'source'}
    transformed=M.transform_row(row,old,new,scale)
    sensor_before=(np.array([160,110])-128)*old[2]*scale/128+old[:2]
    sensor_after=(np.array([transformed['center_x_px'],transformed['center_y_px']])-128)*new[2]*scale/128+new[:2]
    np.testing.assert_allclose(sensor_before,sensor_after,atol=1e-12)


def test_spin_segments_cannot_bridge_mixed_geometry_windows():
    records=[{'segment_id':'S','track_id':'4','segment_frame_offset':i,'segment_length':10,
              'center_x_px':128,'center_y_px':128,'frame_index':i,'timestamp_us':i*500,
              'geometry_continuity_id':int(i>=4),'geometry_window_consistent':i not in [4,5]} for i in range(10)]
    out,dropped=M.eligible_rows(records)
    assert dropped['mixed_geometry_window']==2
    assert [r['frame_index'] for r in out]==[0,1,2,3,6,7,8,9]
    assert len({r['segment_id'] for r in out})==2
    assert all(r['segment_length']==4 for r in out)
