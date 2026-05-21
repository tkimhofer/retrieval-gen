"use client";

import RatsinfoExplorer from "./RatsinfoExplorer";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";

export default function Page() {
  return (
    <div className="h-screen overflow-hidden">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel defaultSize={68} minSize={45}>
          <div className="h-full overflow-auto">
            <RatsinfoExplorer />
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize={32} minSize={22}>
          <iframe
            src="http://localhost:8000"
            className="h-full w-full border-0"
          />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}