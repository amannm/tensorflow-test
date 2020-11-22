import { drawKeypoints, drawSkeleton } from './util.js';

const colors = ['aqua', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white'];

export async function loadAndPredict(config) {
    const img = document.getElementById('image');
    const canvas = document.getElementById('canvas');
    const net = await bodyPix.load({
        architecture: 'ResNet50',
        outputStride: 32,
        quantBytes: 4
    });

    const segmentation = await net.segmentPersonParts(image, {
        flipHorizontal: false,
        internalResolution: mapInternalResolutionConfig(config.internalResolution),
        segmentationThreshold: config.segmentationThreshold
    });

    const coloredPartImage = bodyPix.toColoredPartMask(segmentation);
    const opacity = 0.8;
    const flipHorizontal = false;
    const maskBlurAmount = 0;

    bodyPix.drawMask(
        canvas, img, coloredPartImage, opacity, maskBlurAmount,
        flipHorizontal);

    console.log(segmentation);

    const ctx = canvas.getContext('2d');
    for (let i = segmentation.allPoses.length - 1; i >= 0; i--) {
        const pose = segmentation.allPoses[i];
        const color = colors[i % colors.length];
        drawKeypoints(pose.keypoints, config.poseKeypointThreshold, ctx, 1, color);
        drawSkeleton(pose.keypoints, config.poseKeypointThreshold, ctx, 1, color);
    }

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

export function loadPose(config) {
    const img = document.getElementById('image');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: img.width * config.internalResolution, height: img.height * config.internalResolution },
        quantBytes: 4
    }).then(function (net) {
        const pose = net.estimateSinglePose(img, {
            flipHorizontal: false
        });
        return pose;
    }).then(function (pose) {
        drawKeypoints(pose.keypoints, config.poseKeypointThreshold, ctx, 1, colors[0]);
        drawSkeleton(pose.keypoints, config.poseKeypointThreshold, ctx, 1, colors[0]);
        console.log(pose);
    })
}