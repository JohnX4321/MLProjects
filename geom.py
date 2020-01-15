import jupyter
import pandas,numpy as np,matplotlib,tensorflow,keras


def createParabola(focal_length, centre, rotation):
    t = np.linspace(-math.pi, math.pi,100)
    x_parabola = focal_length * t**2
    y_parabola = 2 * focal_length * t
    if rotation is not None:
        x_parabola, y_parabola = rotateCoordinates(x_parabola, y_parabola, rotation)
    x_parabola = x_parabola + centre[0]
    y_parabola = y_parabola + centre[1]
    return x_parabola, y_parabola

def createCircle(radius, centre):
    theta = np.linspace(0, 2*math.pi,100)
    x_circle = radius * np.cos(theta) + centre[0]
    y_circle = radius * np.sin(theta) + centre[1]
    return x_circle, y_circle

def createEllipse(major_axis, minor_axis, centre, rotation):
    theta = np.linspace(0, 2*math.pi,100)
    x_ellipse = major_axis * np.cos(theta)
    y_ellipse = minor_axis * np.sin(theta)
    if rotation is not None:
        x_ellipse, y_ellipse = rotateCoordinates(x_ellipse,y_ellipse, rotation)
    x_ellipse = x_ellipse + centre[0]
    y_ellipse = y_ellipse + centre[1]
    return x_ellipse, y_ellipse

def createHyperbola(major_axis, conjugate_axis, centre, rotation):
    theta = np.linspace(0, 2*math.pi,100)
    x_hyperbola = major_axis * 1/np.cos(theta) + centre[0]
    y_hyperbola = conjugate_axis * np.tan(theta) + centre[1]
    if rotation is not None:
        x_hyperbola, y_hyperbola = rotateCoordinates(x_hyperbola, y_hyperbola, rotation)
    x_hyperbola = x_hyperbola + centre[0]
    y_hyperbola = y_hyperbola + centre[1]
    return x_hyperbola, y_hyperbola

def rotateCoordinates(x_data, y_data, rot_angle):
    x_ = x_data*math.cos(rot_angle) - y_data*math.sin(rot_angle)
    y_ = x_data*math.sin(rot_angle) + y_data*math.cos(rot_angle)
    return x_,y_


def plotter(x_data, y_data, title):
    fig = plt.figure(figsize=[10,10])
    plt.plot(x_data,y_data,'b--')
    plt.xlabel('X-axis',fontsize=14)
    plt.ylabel('Y-axis',fontsize=14)
    plt.ylim(-18,18)
    plt.xlim(-18,18)
    plt.axhline(y=0, color ="k")
    plt.axvline(x=0, color ="k")
    plt.grid(True)
    saveFile = title + '.svg'
    plt.savefig(saveFile)
    plt.show()

    x, y = createParabola(focal_length=1, centre=[10, 10], rotation=math.pi / 5)
    get_n_samples(x, y, sample_count)


parabola_dataset = pd.DataFrame()
for i in range(1000):
    focal_length = focal_length_array[get_random_index(len(focal_length_array))]
    centre_x = centre_x_arr[get_random_index(len(centre_x_arr))]
    centre_y = centre_y_arr[get_random_index(len(centre_y_arr))]
    rotation = rotation_array[get_random_index(len(rotation_array))]
    x,y = createParabola(focal_length= focal_length, centre= [centre_x, centre_y],rotation= rotation)
    x_, y_ = get_n_samples(x, y, sample_count)
    data = build_dataset(x_, y_, 'parabola')
    parabola_dataset = parabola_dataset.append(data, ignore_index=True)