
const COLORS = ['aqua', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white'];
const LINE_WIDTH = 2;

export async function loadSegmentation(image, internalResolution, segmentationThreshold) {
    const net = await bodyPix.load({
        architecture: 'ResNet50',
        outputStride: 16,
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
    console.log("pose input image size: " + image.width + "x" + image.height)
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 16,
        inputResolution: { width: image.width, height: image.height },
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
        const color = COLORS[i % COLORS.length];
        drawPose(ctx, pose.keypoints, poseKeypointThreshold, color);
    }
}

export function drawPose(ctx, keypoints, poseKeypointThreshold, color = COLORS[0]) {
    drawKeypoints(keypoints, poseKeypointThreshold, ctx, 1, color);
    drawSkeleton(keypoints, poseKeypointThreshold, ctx, 1, color);
}

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = LINE_WIDTH;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawSkeleton(keypoints, minConfidence, ctx, scale = 1, color = COLOR) {
  const adjacentKeyPoints =
    posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  function toTuple({ y, x }) {
    return [y, x];
  }

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
      toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
      scale, ctx);
  });
}

function drawKeypoints(keypoints, minConfidence, ctx, scale = 1, color = COLOR) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const { y, x } = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}