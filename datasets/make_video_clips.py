import os
import logging
import argparse
from math import ceil

import skvideo.io as skvio
import cv2
from moviepy.editor import VideoFileClip


LOGGING_LEVEL = logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        help="Video's frame rate per second"
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        dest='dataset_path',
        help="Dataset's path"
    )
    parser.add_argument(
        '-v',
        '--video-path',
        dest='video_path',
        help="Video's path"
    )
    args = parser.parse_args()
    return args


def get_frames_from_video(video_filename):
    frames = []
    videometadata = skvio.ffprobe(video_filename)
    ffprobe_frame_rate = videometadata['video']['@avg_frame_rate']
    if args.fps is not None:
        frame_rate = str(args.fps)
    else:
        frame_rate = str(ceil(eval(ffprobe_frame_rate)))
    video_reader = skvio.vreader(
        video_filename,
        outputdict={
            '-r': frame_rate
        }
    )
    try:
        for frame in video_reader:
            frames.append(frame)
    except RuntimeError as err:
        logger.warning(
            "Last frame is skipped due to a skvideo bug",
            exc_info=True
        )
    return frames, int(frame_rate)


def main(args):
    logger.info(f"Reading {args.video_path}")
    if not os.path.exists(args.video_path):
        raise IOError(f"Video {args.video_path} not exists")

    frames, frame_rate = get_frames_from_video(args.video_path)
    video = VideoFileClip(args.video_path)
    logger.info("Done")

    frame_idx = 0
    interval = 15
    start_idx = 0
    end_idx = 0
    output_count = 1

    # video.show(frame_idx)
    running = True
    while running:
        frame = frames[frame_idx]
        # Display the resulting frame
        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        ################
        # Key controls #
        ################

        k = cv2.waitKey(25)

        # Move backward
        if k == ord('j'):
            frame_idx = (
                frame_idx - interval
                if frame_idx >= interval
                else frame_idx
            )
            logger.info(frame_idx)

        if k == ord('h'):
            frame_idx = (
                frame_idx - interval * 2
                if frame_idx >= interval * 2
                else frame_idx
            )
            logger.info(frame_idx)

        # Move forward
        elif k == ord('k'):
            frame_idx = (
                frame_idx + interval
                if frame_idx < len(frames) - interval
                else frame_idx
            )
            logger.info(frame_idx)

        elif k == ord('l'):
            frame_idx = (
                frame_idx + interval * 2
                if frame_idx < len(frames) - interval * 2
                else frame_idx
            )
            logger.info(frame_idx)

        elif k == ord('d'):
            start_idx = frame_idx
            logger.info(f"Chang start: {start_idx}")

        elif k == ord('f'):
            end_idx = frame_idx
            logger.info(f"Change end: {end_idx}")

        elif k == ord('s'):
            if start_idx >= end_idx:
                continue
            video_clip = video.subclip(
                start_idx / frame_rate,
                end_idx / frame_rate
            )
            video_clip.preview()
            video_dir = "".join(args.video_path.split('.mp4')[:-1])
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            output_filename = f"{video_dir}/{output_count:03}.mp4"
            while os.path.exists(output_filename):
                output_count += 1
                output_filename = f"{video_dir}/{output_count:03}.mp4"
            video_clip.write_videofile(output_filename)

        # Exit
        elif k == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
