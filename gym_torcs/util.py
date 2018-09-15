import os
import time
import smtplib

def combine_dbs(root):
    """ Walk through a directory tree and combine all 'db.csv' files into one."""
    assert os.path.exists(root)
    master = []
    for root, dirs, files in os.walk(root):
        for f in files:
            if f=="db.csv":
                with open(os.path.join(root, f), 'r') as fd:
                    text = fd.readlines()
                    master += text[1:]
    with open("./master.csv", 'w') as f:
        f.writelines(master)

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]



def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

def get_secret(sfile='./.secret'):
    with open(sfile,'r') as f:
        secret = f.read()
    return secret

def email_when_finished(func):
    def wrapped_func(*args, **kwargs):
        tic = time.time()
        error = None
        try:
            func(*args, **kwargs)
        except Exception as e:
            error = e
        toc = time.time()
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        secret = get_secret()
        server.login("sdcc@umn.edu", secret)
        msg = "Training finished. Took %s" % str(toc-tic)
        server.sendmail("sdcc@umn.edu", "bittn037@umn.edu", msg)
        server.quit()


    return wrapped_func
