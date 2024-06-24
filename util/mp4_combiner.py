# Ensure you're using the latest version of Pillow
from moviepy.editor import VideoFileClip, concatenate_videoclips


def main():
    # List of video file paths
    video_files = [f"videos/output_video_{i}.mp4" for i in range(7)]

    # Desired resolution and frame rate
    target_fps = 60.0

    # Load and process all video clips
    clips = []
    for video in video_files:
        clip = VideoFileClip(video)
        clip = clip.set_fps(target_fps)
        clips.append(clip)

    # Concatenate the video clips
    final_clip = concatenate_videoclips(clips)

    # Write the final video to a file
    final_clip.write_videofile("output_video.mp4")


if __name__ == "__main__":
    main()
