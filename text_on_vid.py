import moviepy.editor as mp
my_vid=mp.VideoFileClip("",audio=True)
w,h=mvsize=my_vid.size

my_text=mp.TextClip("ART ADD",font="Amiri-regular",color='white',fontsize=34)
text_col=my_text.on_color(size=(w+my_text.w,my_text.h+50),color=(0,0,0),pos=(6,'center'),col_opacity=0.6)
txt_mov=text_col.set_position(lambda t:(max(w/30,int(w-0.5*w*t)),max(5*h/,int(100*t))))
final=mp.CompositeVideoClip([my_vid,my_text])
final.subclip(0,17).write_videofile("",fps=24,codec='libx264')
