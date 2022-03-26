import os
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd

REPLAY_DIR = "D:\\Videos"
FRAME_RATE = 48


def segment_circle(num_segments: int):
    """Split a circle into num_segments segments

    Args:
        num_segments (int): number of segments to split the circle into

    Returns:
        a numpy array of size [num_segments x 3] containing the (x, y)
        co-ordinates of the segment and it's angle in radians
    """
    # calculate the size in radians of each segment of the circle
    segment_rad = 2 * np.pi / num_segments
    # create a list of all the radians for each segment
    segment_rads = segment_rad * np.arange(num_segments)
    # calculate the X & Y co-ordinates for each segment
    x_cords = np.cos(segment_rads)
    y_cords = np.sin(segment_rads)
    # return the concatenation of the 3 arrays along the second axis
    return np.c_[x_cords, y_cords, segment_rads]


def create_video():
    """Creates the animation video."""
    r = 7.0

    # months - these go backwards and start at March
    months = ["Mar", "Feb", "Jan", "Dec", "Nov", "Oct", "Sep", "Aug", "Jul", "Jun", "May", "Apr"]
    # month index lookup table
    month_idx = [2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    radius = r + 0.4
    month_points = segment_circle(len(months))
    df = pd.read_csv('D:/Data/ClimateData/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv')
    df['Time'] = pd.to_datetime(df['Time'])  # convert the Time column to a DateTime
    r_factor = r / 3.6  # scale goes from -1.5 to 2.1
    x_orig = df['Anomaly (deg C)'].to_numpy() + 1.5
    x_vals = []
    y_vals = []
    for i in range(0, len(x_orig)):
        r_pos = x_orig[i] * r_factor
        x_unit_r, y_unit_r = month_points[month_idx[i % 12], :2]
        x_r, y_r = (r_pos * x_unit_r, r_pos * y_unit_r)
        x_vals.append(x_r)
        y_vals.append(y_r)

    ffmpeg = Popen(['ffmpeg',
                    '-y',  # overwrite output files without asking
                    '-f', 'image2pipe',  # use a pipe for the input
                    '-c:v', 'png',  # use the png encoder
                    '-s', '640x480',  # screen dimensions
                    '-pix_fmt', 'rgba',  # pixel format of RGBA
                    '-framerate', str(FRAME_RATE),  # frame rate
                    '-i', '-',  # input is from the pipe
                    '-c:v', 'libx264',  # use the libx264 codec - H264
                    '-b:v', '4M',  # bitrate
                    '-pix_fmt', 'yuv420p',  # use the yuv420p pixel format
                    os.path.join(REPLAY_DIR, "climate_spiral.mp4")],
                   stdin=PIPE,
                   stderr=STDOUT)

    fig, ax = plt.subplots(figsize=(14, 14))
    for i in range(len(x_vals)):
        ax.clear()
        fig.patch.set_facecolor('gray')
        ax.axis('equal')

        ax.set(xlim=(-10, 10), ylim=(-10, 10))

        circle = plt.Circle((0, 0), r, fc='#000000')
        ax.add_patch(circle)

        circle_2 = plt.Circle((0, 0), r_factor * 2.5, ec='red', fc=None, fill=False, lw=3.0)
        ax.add_patch(circle_2)
        circle_1_5 = plt.Circle((0, 0), r_factor * 3.0, ec='red', fc=None, fill=False, lw=3.0)
        ax.add_patch(circle_1_5)

        props_months = {'ha': 'center', 'va': 'center', 'fontsize': 24, 'color': 'white'}
        props_year = {'ha': 'center', 'va': 'center', 'fontsize': 36, 'color': 'white'}
        props_temp = {'ha': 'center', 'va': 'center', 'fontsize': 32, 'color': 'red'}
        ax.text(0, r_factor * 2.5, '1.5°C', props_temp, bbox=dict(facecolor='black'))
        ax.text(0, r_factor * 3.0, '2.0°C', props_temp, bbox=dict(facecolor='black'))
        ax.text(0, r + 1.4, 'Global temperature change (1850-2021)', props_year)
        year = 1850 + i // 12

        # draw the month legends around the rim of the circle
        for j in range(0, len(months)):
            x_unit_r, y_unit_r, angle = month_points[j]
            x_radius, y_radius = (radius * x_unit_r, radius * y_unit_r)
            angle = angle - 0.5 * np.pi
            ax.text(x_radius, y_radius, months[j], props_months, rotation=np.rad2deg(angle), )

        if i > 1:
            x_seg = x_vals[:i]
            y_seg = y_vals[:i]
            pts = np.array([x_seg, y_seg]).T.reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, 3.6))
            lc.set_array(np.asarray(x_orig))
            # add all the lines with their colours
            plt.gca().add_collection(lc)
            ax.text(0, 0, str(year), props_year)
        # rescale everything so it fits in the space
        ax.autoscale()
        # turn off the graph axis
        ax.axis("off")
        plt.savefig(ffmpeg.stdin, format='png')

    ffmpeg.stdin.close()
    ffmpeg.wait()


def main():
    """Main function."""
    create_video()
    print('Finished!')


if __name__ == '__main__':
    main()
