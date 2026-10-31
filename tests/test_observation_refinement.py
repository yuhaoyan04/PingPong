import cv2
import numpy as np
import pytest
from pingpong_event_pipeline.observation_refinement import (
    fit_boundary_circle,register_marker,registration_image,edge_sectors)


def ring():
    rng=np.random.default_rng(7);theta=np.linspace(0,2*np.pi,3000,endpoint=False)
    r=35+rng.normal(0,.25,len(theta))
    return np.array([100.,100.])+np.c_[np.cos(theta)*r,np.sin(theta)*r]


def test_noisy_circle_corrects_center_and_radius_on_unused_arcs():
    points=ring();noise=np.random.default_rng(3).uniform(55,145,(1000,2))
    fit=fit_boundary_circle(np.r_[points,noise],[101.5,98.5,36])
    assert fit['accepted']
    np.testing.assert_allclose(fit['circle'],[100,100,35],atol=.15)
    assert fit['audit_after_median_px']<fit['audit_before_median_px']*.2


def test_dense_repeated_arc_cannot_dominate_boundary_fit():
    points=ring();arc=points[:400]
    a=fit_boundary_circle(points,[101.5,98.5,36])
    b=fit_boundary_circle(np.r_[points,np.tile(arc,(40,1))],[101.5,98.5,36])
    assert a['accepted'] and b['accepted']
    np.testing.assert_allclose(a['circle'],b['circle'],atol=.15)


def test_short_arc_or_random_activity_does_not_create_a_valid_ball():
    short=fit_boundary_circle(ring()[:200],[101.5,98.5,36])
    assert not short['accepted']
    random=fit_boundary_circle(np.random.default_rng(11).uniform(55,145,(1500,2)),[100,100,35])
    assert not random['accepted']
    empty=fit_boundary_circle([], [100,100,35])
    assert not empty['accepted'] and empty['circle']==[100.,100.,35.]


def texture():
    rng=np.random.default_rng(13)
    image=np.full((128,128),128,np.uint8)
    for x,y,c in zip(rng.integers(38,90,140),rng.integers(38,90,140),rng.choice([20,240],140)):
        cv2.circle(image,(int(x),int(y)),1,int(c),-1)
    return cv2.GaussianBlur(image,(3,3),.55)


def test_affine_anchor_tracks_fixed_texture_despite_bbox_midpoint_shift():
    reference=texture();matrix=cv2.getRotationMatrix2D((64,64),6,1.04);matrix[:,2]+=[2.5,-1.5]
    current=cv2.warpAffine(reference,matrix,(128,128),borderValue=128)
    physical=matrix[:,:2]@np.array([64,64])+matrix[:,2]
    bbox_center=physical+[3,-2]
    fit=register_marker(reference,current,[35,35,93,93],[64,64],bbox_center)
    assert fit['accepted'],fit
    assert np.linalg.norm(np.array(fit['center'])-physical)<.25
    assert fit['audit_after_median_px']<fit['audit_before_median_px']*.25


def test_blank_marker_cannot_be_localized_from_prediction_alone():
    blank=np.full((128,128),128,np.uint8)
    fit=register_marker(blank,blank,[35,35,93,93],[64,64],[65,63])
    assert not fit['accepted'] and fit['center']==[65.,63.]


def test_registration_is_polarity_sensitive_without_shifting_coordinates():
    rgb=np.zeros((32,32,3),np.uint8);rgb[10:15,8:12,2]=255;rgb[10:15,20:24,:2]=255
    image=registration_image(rgb)
    assert image[12,10]<128 and image[12,22]>128 and image[3,3]==128
    with pytest.raises(ValueError):registration_image(np.zeros((32,32)))
    with pytest.raises(ValueError):edge_sectors([[1,2]],[0,0,0])
