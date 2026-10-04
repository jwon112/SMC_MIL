#!/usr/bin/env python3
"""Read-only source WSI audit for the two v3 review cases; separate output only.

No custom DicomPyramidReader, stain processing, resampling, or training.
Frame mapping: DICOM PS3.3 C.7.6.17.3, C.8.12.14.
https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.3.html
https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.pixel_array.html
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

CASES=[('S2039743__S20-39743-L-PLHE_1',901),('S2039743__S20-39743-C-PLHE_0',26997)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def scalar(v):
    if isinstance(v,dict):return {str(k):scalar(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [scalar(x) for x in v]
    if isinstance(v,np.generic):return scalar(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    if v is None or isinstance(v,(str,int,float,bool)):return v
    return str(v)


def write_json(path,value):
    Path(path).write_text(json.dumps(scalar(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def positive_int(ds,name):
    value=getattr(ds,name,None)
    if value is None or int(value)<=0 or float(value)!=int(value):
        raise ValueError('Missing/invalid '+name)
    return int(value)


def header_summary(ds):
    keys=['SOPClassUID','ImageType','Rows','Columns','NumberOfFrames','SamplesPerPixel',
          'PhotometricInterpretation','PlanarConfiguration','BitsAllocated','BitsStored',
          'PixelRepresentation','TotalPixelMatrixColumns','TotalPixelMatrixRows',
          'TotalPixelMatrixFocalPlanes','NumberOfOpticalPaths','DimensionOrganizationType',
          'ImagedVolumeWidth','ImagedVolumeHeight','ImagedVolumeDepth','PixelSpacing',
          'LossyImageCompression','LossyImageCompressionRatio','LossyImageCompressionMethod']
    result={k:scalar(getattr(ds,k,None)) for k in keys}
    result['TransferSyntaxUID']=str(getattr(getattr(ds,'file_meta',None),'TransferSyntaxUID',''))
    result['per_frame_count']=len(getattr(ds,'PerFrameFunctionalGroupsSequence',[]) or [])
    result['optical_path_count']=len(getattr(ds,'OpticalPathSequence',[]) or [])
    result['has_concatenation']=bool(getattr(ds,'ConcatenationUID',None))
    icc=getattr(ds,'ICCProfile',None)
    result['icc_profile_present']=bool(icc) or any(bool(getattr(p,'ICCProfile',None)) for p in getattr(ds,'OpticalPathSequence',[]) or [])
    result['privacy']='Whitelisted technical fields only; no patient header dump.'
    return result


def tiles_for_roi(ds,x,y,width,height,max_frame_pixels=16_000_000):
    """Return (frame_index, zero-based x,y) without inferring missing tile order."""
    cols=positive_int(ds,'TotalPixelMatrixColumns');rows=positive_int(ds,'TotalPixelMatrixRows')
    tw=positive_int(ds,'Columns');th=positive_int(ds,'Rows');nf=positive_int(ds,'NumberOfFrames')
    if min(x,y)<0 or min(width,height)<=0 or x+width>cols or y+height>rows:
        raise ValueError('ROI exceeds selected total matrix')
    if tw*th>max_frame_pixels:
        raise ValueError(f'One decoded frame has {tw*th:,} pixels, exceeds safety limit; OpenSlide may still work')
    if getattr(ds,'ConcatenationUID',None):raise ValueError('Concatenated instances not supported; do not guess frame offsets')
    if int(getattr(ds,'TotalPixelMatrixFocalPlanes',1))!=1 or int(getattr(ds,'NumberOfOpticalPaths',1))!=1 or len(getattr(ds,'OpticalPathSequence',[]) or [])>1:
        raise ValueError('Multiple focal planes/optical paths require explicit plane selection')
    image_type=getattr(ds,'ImageType',[]) or []
    if isinstance(image_type,str):image_type=image_type.split('\\')
    if any(str(v) in {'LABEL','OVERVIEW','THUMBNAIL','LOCALIZER'} for v in image_type):
        raise ValueError('Selected instance is not a volume layer')
    groups=getattr(ds,'PerFrameFunctionalGroupsSequence',[]) or []
    # TILED_FULL may carry other per-frame macros while omitting all positions.
    if groups and str(getattr(ds,'DimensionOrganizationType',''))=='TILED_FULL' and all(not (getattr(g,'PlanePositionSlideSequence',[]) or []) for g in groups):
        if len(groups)!=nf:raise ValueError('Incomplete per-frame metadata')
        groups=[]
    tiles=[];zs=set();paths=set()
    if groups:
        if len(groups)!=nf:raise ValueError('Incomplete per-frame metadata')
        for i,g in enumerate(groups):
            positions=getattr(g,'PlanePositionSlideSequence',[]) or []
            if len(positions)!=1:raise ValueError('Missing/ambiguous explicit frame position')
            pos=positions[0]
            tx=positive_int(pos,'ColumnPositionInTotalImagePixelMatrix')-1
            ty=positive_int(pos,'RowPositionInTotalImagePixelMatrix')-1
            if tx>=cols or ty>=rows:raise ValueError('Frame origin outside total matrix')
            if hasattr(pos,'ZOffsetInSlideCoordinateSystem'):zs.add(float(pos.ZOffsetInSlideCoordinateSystem))
            for o in getattr(g,'OpticalPathIdentificationSequence',[]) or []:
                paths.add(str(getattr(o,'OpticalPathIdentifier','')))
            if tx<x+width and tx+tw>x and ty<y+height and ty+th>y:tiles.append((i,tx,ty))
        if len(zs)>1 or len(paths)>1:raise ValueError('Multiple explicit z/optical paths; ambiguous ROI')
        method='explicit_per_frame_1based_to_0based'
    elif str(getattr(ds,'DimensionOrganizationType',''))=='TILED_FULL':
        nx=(cols+tw-1)//tw;ny=(rows+th-1)//th
        if nf!=nx*ny:raise ValueError('TILED_FULL frame count differs from one complete plane')
        for ty in range(y//th,(y+height-1)//th+1):
            for tx in range(x//tw,(x+width-1)//tw+1):tiles.append((ty*nx+tx,tx*tw,ty*th))
        method='TILED_FULL_single_plane_row_major'
    elif nf==1 and (tw,th)==(cols,rows):
        tiles=[(0,0,0)];method='single_frame_equals_total_matrix'
    else:
        raise ValueError('No explicit frame positions or TILED_FULL: tile order cannot be assumed')
    if len(tiles)>64:raise ValueError('ROI requires >64 frames; refusing unexpected workload')
    coverage=np.zeros((height,width),np.uint8)
    for _,tx,ty in tiles:
        l=max(x,tx)-x;t=max(y,ty)-y;r=min(x+width,tx+tw)-x;b=min(y+height,ty+th)-y
        if coverage[t:b,l:r].any():raise ValueError('Overlapping frames or multiple planes in ROI')
        coverage[t:b,l:r]=1
    if not coverage.all():raise ValueError('ROI contains unencoded/missing pixels; no fabricated white filling')
    return tiles,method


def assemble_roi(ds,tiles,decode,x,y,width,height):
    out=np.empty((height,width,3),np.uint8)
    for i,tx,ty in tiles:
        a=np.asarray(decode(i))
        if a.dtype!=np.uint8 or a.shape!=(int(ds.Rows),int(ds.Columns),3):
            raise ValueError(f'Unsupported frame shape/dtype: {a.shape}/{a.dtype}; no min-max normalization')
        l=max(x,tx);t=max(y,ty);r=min(x+width,tx+int(ds.Columns));b=min(y+height,ty+int(ds.Rows))
        out[t-y:b-y,l-x:r-x]=a[t-ty:b-ty,l-tx:r-tx]
    return out


def compare(a,b):
    if a.shape!=b.shape:raise ValueError('Comparison shape mismatch')
    diff=np.abs(a.astype(float)-b.astype(float))
    return dict(mae=float(diff.mean()),max_absolute_error=float(diff.max()),
                exact_pixel_fraction=float(np.mean(np.all(a==b,axis=2))))


def pillow_jpeg_frames(path,indices,ds,max_compressed_bytes):
    """Separate JPEG decoder fallback; never decode the whole WSI array."""
    import pydicom
    uid=str(ds.file_meta.TransferSyntaxUID)
    if uid!='1.2.840.10008.1.2.4.50':
        raise ValueError('Pillow fallback is restricted to 8-bit JPEG Baseline DICOM')
    if path.stat().st_size>max_compressed_bytes:
        raise ValueError('Compressed-file safety limit exceeded for Pillow fallback')
    whole=pydicom.dcmread(str(path))
    # Public encapsulation APIs across pydicom 2.x/3.x; no hand-written JPEG slicing.
    try:
        from pydicom.encaps import generate_frames
        ext=None
        if hasattr(whole,'ExtendedOffsetTable') and hasattr(whole,'ExtendedOffsetTableLengths'):
            ext=(whole.ExtendedOffsetTable,whole.ExtendedOffsetTableLengths)
        frames=generate_frames(whole.PixelData,number_of_frames=int(ds.NumberOfFrames),extended_offsets=ext)
    except ImportError:
        from pydicom.encaps import generate_pixel_data_frame
        frames=generate_pixel_data_frame(whole.PixelData,nr_frames=int(ds.NumberOfFrames))
    wanted=set(indices);found={}
    for i,frame in enumerate(frames):
        if i in wanted:
            with Image.open(io.BytesIO(frame)) as im:
                if im.size!=(int(ds.Columns),int(ds.Rows)) or im.mode not in {'RGB','YCbCr'}:
                    raise ValueError('Unexpected JPEG size/color mode')
                found[i]=np.array(im.convert('RGB'))
        if wanted.issubset(found):break
    if set(found)!=wanted:raise ValueError('Requested JPEG frames not found')
    return found


def openslide_roi(path,expected,x,y,size):
    import openslide
    with openslide.OpenSlide(str(path)) as slide:
        if tuple(slide.dimensions)!=tuple(expected):
            raise ValueError(f'OpenSlide level-0 dimensions differ: {slide.dimensions} vs {expected}')
        region=np.asarray(slide.read_region((x,y),0,(size,size)))
        if region.shape!=(size,size,4) or not (region[...,3]==255).all():
            raise ValueError('OpenSlide ROI includes missing/transparent pixels')
        info=dict(version=getattr(openslide,'__version__','unknown'),
                  library_version=getattr(openslide,'__library_version__','unknown'),
                  vendor=slide.properties.get('openslide.vendor'),dimensions=list(slide.dimensions),
                  icc_transform_applied=False)
        return region[...,:3].copy(),info


def diagnose_case(packet,out,max_frame_pixels,max_compressed_bytes):
    meta=json.loads((packet/'review.json').read_text())
    source=[Path(p) for p in meta['source_paths']]
    if meta['source']!='dicom' or len(source)!=1:raise ValueError('Requires one selected DICOM instance')
    path=source[0];x=int(meta['x_level0']);y=int(meta['y_level0']);size=int(meta['patch_size'])
    if not 1<=size<=2048:raise ValueError('Unexpected ROI size')
    if sha(packet/'raw_core.png')!=meta['file_sha256']['raw_core.png']:
        raise ValueError('Saved original patch checksum mismatch')
    with Image.open(packet/'raw_core.png') as im:reference=np.array(im.convert('RGB'))
    if reference.shape!=(size,size,3):raise ValueError('Reference must be unresized core')
    out.mkdir();Image.fromarray(reference).save(out/'reference_reader_core.png')
    report=dict(slide_id=meta['slide_id'],patch_index=meta['patch_index'],x_level0=x,y_level0=y,
        size=size,source_path=str(path),reference_sha256=sha(packet/'raw_core.png'),
        source_stat=dict(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns),
        independent_viewer_verified=False,clinical_quality_validated=False,
        note='Alternate mapping/decoder checks only. Agreement cannot prove focus, clinical validity, or correct scanner acquisition. No ICC transform or contrast normalization.',backends={})
    images={'saved_custom_reader':reference};ds=None;tiles=None
    try:
        import pydicom
        report['pydicom_version']=pydicom.__version__
        ds=pydicom.dcmread(str(path),stop_before_pixels=True)
        report['header']=header_summary(ds)
        if (positive_int(ds,'TotalPixelMatrixColumns'),positive_int(ds,'TotalPixelMatrixRows'))!=tuple(meta['reader_dimensions']):
            raise ValueError('Header matrix differs from original reader dimensions')
        if positive_int(ds,'BitsAllocated')!=8 or positive_int(ds,'BitsStored')!=8 or positive_int(ds,'SamplesPerPixel')!=3 or int(getattr(ds,'PixelRepresentation',0))!=0:
            raise ValueError('This diagnostic requires unsigned 8-bit three-channel color data')
        if str(getattr(ds,'PhotometricInterpretation','')) not in {'RGB','YBR_FULL','YBR_FULL_422'}:
            raise ValueError('Unsupported color interpretation; do not silently treat non-RGB decoder output as RGB')
        tiles,method=tiles_for_roi(ds,x,y,size,size,max_frame_pixels)
        report['frame_mapping']=dict(method=method,frames=[dict(index=i,x=tx,y=ty) for i,tx,ty in tiles])
    except Exception as exc:
        report['frame_mapping_error']=f'{type(exc).__name__}: {exc}'
        tiles=None
    def run_backend(name,fn):
        print(f'  {name}...',flush=True)
        try:
            a,info=fn()
            if a.dtype!=np.uint8 or a.shape!=reference.shape:raise ValueError('RGB output dtype/shape mismatch')
            metrics=compare(reference,a)
            Image.fromarray(a).save(out/f'{name}_core.png');images[name]=a
            report['backends'][name]=dict(status='decoded',comparison_to_saved_reader=metrics,**info)
            print(f'  {name}: MAE={metrics["mae"]:.6f}; exact pixels={metrics["exact_pixel_fraction"]:.3%}',flush=True)
        except Exception as exc:
            report['backends'][name]=dict(status='unavailable_or_rejected',error=f'{type(exc).__name__}: {exc}')
            print(f'  {name}: {type(exc).__name__}: {exc}',flush=True)
    # OpenSlide remains an alternative even if Python frame metadata is incomplete.
    run_backend('openslide',lambda:openslide_roi(path,meta['reader_dimensions'],x,y,size))
    if tiles is not None:
        def pydicom_decode():
            from pydicom.pixels import pixel_array
            return assemble_roi(ds,tiles,lambda i:pixel_array(str(path),index=i,raw=False),x,y,size,size),dict(icc_transform_applied=False,frame_indices=[i for i,_,_ in tiles])
        run_backend('pydicom_frames',pydicom_decode)
        def pillow_decode():
            frames=pillow_jpeg_frames(path,[i for i,_,_ in tiles],ds,max_compressed_bytes)
            return assemble_roi(ds,tiles,frames.__getitem__,x,y,size,size),dict(pillow_version=pillow_version,icc_transform_applied=False)
        run_backend('pillow_jpeg',pillow_decode)
    else:
        for name in ['pydicom_frames','pillow_jpeg']:
            report['backends'][name]=dict(status='not_attempted',error='Frame mapping unresolved; no guessed grid')
    report['pairwise_comparisons']={}
    names=list(images)
    for i,name in enumerate(names):
        for other in names[i+1:]:report['pairwise_comparisons'][name+' vs '+other]=compare(images[name],images[other])
    canvas=Image.new('RGB',(len(images)*size,size+32),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,a) in enumerate(images.items()):
        canvas.paste(Image.fromarray(a),(i*size,32));draw.text((i*size+4,8),name,fill='black')
    canvas.save(out/'comparison.png')
    report['alternate_paths_decoded']=len(images)-1
    if len(images)==1:report['conclusion']='unresolved_no_alternate_decode'
    elif all(compare(reference,a)['mae']==0 for a in list(images.values())[1:]):
        report['conclusion']='exact_agreement_in_available_paths_not_a_focus_validation'
    else:report['conclusion']='pixel_differences_require_spatial_and_color_review'
    report['file_sha256']={p.name:sha(p) for p in out.glob('*.png')}
    write_json(out/'diagnostic.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root',type=Path,default=Path('/home/jupyter/data/image_team/pathomics_v3_measurementqa/smoke'))
    p.add_argument('--output',type=Path,default=Path('/home/jupyter/data/image_team/pathomics_v3_measurementqa/dicom_pixel_audit'))
    p.add_argument('--max-frame-pixels',type=int,default=16_000_000)
    p.add_argument('--max-compressed-mib',type=int,default=512)
    a=p.parse_args()
    if not 1<=a.max_frame_pixels<=64_000_000 or not 1<=a.max_compressed_mib<=1024:
        p.error('Safety bounds: frame pixels 1..64M, compressed MiB 1..1024')
    # Refuse existing output to preserve prior diagnostics and user data.
    a.output.mkdir(parents=True,exist_ok=False)
    results=[];start=time.monotonic()
    for sid,index in CASES:
        print(f'[CASE] {sid}, patch {index}',flush=True)
        packet=a.review_root/'slides'/sid/'measurement_review'/f'{index:08d}'
        try:results.append(diagnose_case(packet,a.output/f'{sid}_patch{index}',a.max_frame_pixels,a.max_compressed_mib*1024**2))
        except Exception as exc:
            results.append(dict(slide_id=sid,patch_index=index,status='failed',error=f'{type(exc).__name__}: {exc}'))
            print(f'  FAILED: {exc}',flush=True)
    summary=dict(python=platform.python_version(),numpy=np.__version__,pillow=pillow_version,
        script_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start,results=results,
        source_data_modified=False,automatic_quality_exclusion=False,
        caution='Missing codecs/metadata are diagnostic failures, not evidence of poor image quality. Backends may share codec libraries; agreement is not independent pathological validation.')
    write_json(a.output/'summary.json',summary)
    print(f'Saved: {a.output}',flush=True)
    for r in results:print(r['slide_id'],r.get('conclusion',r.get('status')),flush=True)


if __name__=='__main__':main()
