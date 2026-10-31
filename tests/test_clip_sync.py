import pytest

from pingpong_event_pipeline.clip_sync import exact_pairs, manifest_map


def row(frame,t):
    return {'frame_index':str(frame),'timestamp_us':str(t),'image':f'binary/{frame}.png'}


def test_exact_timestamp_pairing_does_not_align_frame_zero_or_shift_a_camera():
    result=exact_pairs([row(0,100),row(1,200),row(2,300)], [row(0,200),row(1,300)])
    assert [p['timestamp_us'] for p in result]==[200,300]
    assert [(p['F3E_frame_index'],p['F59_frame_index']) for p in result]==[(1,0),(2,1)]
    assert all(p['F3E_timestamp_us']==p['F59_timestamp_us'] for p in result)


def test_no_overlap_is_not_converted_into_a_fake_synchronized_pair():
    assert exact_pairs([row(0,100)],[row(0,200)])==[]


@pytest.mark.parametrize('rows',[[row(0,100),row(1,100)],[row(0,200),row(1,100)]])
def test_ambiguous_or_reversed_timestamp_manifests_are_rejected(rows):
    with pytest.raises(ValueError,match='timestamp'):
        manifest_map(rows)
