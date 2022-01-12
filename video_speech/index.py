import speech_recognition as sr
import moviepy.editor as mp

vid_path=""
clip=mp.VideoClip(vid_path)
clip.audio.write_audiofile(r"conv.wav")
r=sr.Recognizer()
audio=sr.AudioFile("conv.wav")
with audio as src:
    audio_file=r.record(src)
res=r.recognize_google(audio_file)

