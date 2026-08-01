import queue
import logging
from typing import Optional
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)

class LogIngestRingBuffer:
    """Thread-safe high-throughput ring buffer emulating a Kafka topic ingestion partition.
    
    Provides non-blocking writes and a blocking consumer queue to buffer streaming log flows.
    """
    def __init__(self, max_capacity: int = 50000):
        self.max_capacity = max_capacity
        self.queue = queue.Queue(maxsize=max_capacity)
        self.dropped_events_count = 0

    def put(self, flow: CanonicalFlow) -> bool:
        """Pushes a canonical flow record onto the buffer.
        
        If the capacity is exceeded, it discards the oldest record (behaving as a ring-buffer)
        and increments the drop counter.
        """
        try:
            # Try to append without blocking
            self.queue.put_nowait(flow)
            return True
        except queue.Full:
            # Ring buffer behaviour: drop oldest event to make space
            try:
                self.queue.get_nowait()
                self.dropped_events_count += 1
            except queue.Empty:
                pass
            
            try:
                self.queue.put_nowait(flow)
                return True
            except queue.Full:
                return False

    def get(self, timeout: Optional[float] = None) -> Optional[CanonicalFlow]:
        """Pulls the next CanonicalFlow event from the ingest buffer.
        
        Blocks if queue is empty until an item is available or timeout occurs.
        """
        try:
            return self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        """Returns the current number of buffered flows."""
        return self.queue.qsize()

    def get_dropped_count(self) -> int:
        """Returns the total number of events dropped due to capacity overflow."""
        return self.dropped_events_count
