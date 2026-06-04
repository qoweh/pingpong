import { Pause, Play, RotateCcw } from "lucide-react";

import type { PlaybackState } from "../simulation/types";

interface PlaybackControlsProps {
  playback: PlaybackState;
  onPlaybackChange: (playback: PlaybackState) => void;
  onReset: () => void;
}

export function PlaybackControls({ playback, onPlaybackChange, onReset }: PlaybackControlsProps) {
  return (
    <div className="toolbar" aria-label="Playback controls">
      <button
        className={playback === "playing" ? "icon-button active" : "icon-button"}
        type="button"
        title="Play"
        aria-label="Play"
        onClick={() => onPlaybackChange("playing")}
      >
        <Play size={18} />
      </button>
      <button
        className={playback === "paused" ? "icon-button active" : "icon-button"}
        type="button"
        title="Pause"
        aria-label="Pause"
        onClick={() => onPlaybackChange("paused")}
      >
        <Pause size={18} />
      </button>
      <button className="icon-button" type="button" title="Reset" aria-label="Reset" onClick={onReset}>
        <RotateCcw size={18} />
      </button>
    </div>
  );
}
