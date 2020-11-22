import { drawKeypoints, drawSkeleton, getInputSize } from './util.js';

const colors = ['aqua', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white'];

export async function loadSegmentation(image, internalResolution, segmentationThreshold) {
    const net = await bodyPix.load({
        architecture: 'ResNet50',
        outputStride: 32,
        quantBytes: 4
    });
    return await net.segmentPersonParts(image, {
        flipHorizontal: false,
        internalResolution: mapInternalResolutionConfig(internalResolution),
        segmentationThreshold: segmentationThreshold
    });
}

function mapInternalResolutionConfig(internalResolution) {
    switch (internalResolution) {
        case 0.25:
            return 'low';
        case 0.5:
            return 'medium';
        case 0.75:
            return 'high';
        case 1.0:
            return 'full';
        default:
            throw 'unsupported internal resolution setting: ' + internalResolution;
    }
}

export async function loadPose(image, internalPoseResolution) {
    const [width, height] = calculateDimensions(image, internalPoseResolution)
    console.log("pose input image size: " + width + "x" + height)
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: width, height: height },
        quantBytes: 4
    });
    return await net.estimateSinglePose(image, {
        flipHorizontal: false
    });
}
export function drawSegmentation(canvas, image, segmentation) {
    const coloredPartImage = bodyPix.toColoredPartMask(segmentation);
    const opacity = 0.8;
    const flipHorizontal = false;
    const maskBlurAmount = 0;
    bodyPix.drawMask(
        canvas, image, coloredPartImage, opacity, maskBlurAmount,
        flipHorizontal);
}

export function drawPoses(ctx, allPoses, poseThreshold, poseKeypointThreshold) {
    for (let i = allPoses.length - 1; i >= 0; i--) {
        const pose = allPoses[i];
        if (pose.score < poseThreshold) {
            continue;
        }
        const color = colors[i % colors.length];
        drawPose(ctx, pose.keypoints, poseKeypointThreshold, color);
    }
}

export function drawPose(ctx, keypoints, poseKeypointThreshold, color = colors[0]) {
    drawKeypoints(keypoints, poseKeypointThreshold, ctx, 1, color);
    drawSkeleton(keypoints, poseKeypointThreshold, ctx, 1, color);
}

function calculateDimensions(image, overrideScalingFactor) {
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    if (overrideScalingFactor > 0) {
        const imageWidth = Math.floor(width * overrideScalingFactor);
        const imageHeight = Math.floor(height * overrideScalingFactor);
        return [imageWidth, imageHeight];
    } else {
        const defaultScalingFactor = Math.sqrt(345600/(width*height));
        const imageWidth = Math.floor(width * defaultScalingFactor);
        const imageHeight = Math.floor(height * defaultScalingFactor);
        return [imageWidth, imageHeight];
    }
}