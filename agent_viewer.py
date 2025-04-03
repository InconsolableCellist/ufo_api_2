import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import json
from datetime import datetime
import os
import threading
import time
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class AgentViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure main window
        self.title("Agent State Viewer")
        self.geometry("1200x800")
        
        # Initialize data paths
        self.memory_path = "agent_memory.pkl"
        self.tool_registry_path = "tool_registry_state.pkl"
        self.journal_path = "agent_journal.txt"
        
        # Initialize data storage
        self.memory_data = None
        self.tool_registry_data = None
        self.journal_data = None
        
        # Add cache for processed memories
        self.memory_cache = {
            'long_term': [],
            'short_term': [],
            'last_update': None
        }
        
        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(expand=True, fill='both')
        
        # Create tabs
        self.memory_tab = ttk.Frame(self.notebook)
        self.emotion_tab = ttk.Frame(self.notebook)
        self.emotion_graph_tab = ttk.Frame(self.notebook)
        self.goals_tab = ttk.Frame(self.notebook)
        self.tools_tab = ttk.Frame(self.notebook)
        self.personality_tab = ttk.Frame(self.notebook)
        self.journal_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.memory_tab, text='Memory Browser')
        self.notebook.add(self.emotion_tab, text='Emotional State')
        self.notebook.add(self.emotion_graph_tab, text='Emotion Graph')
        self.notebook.add(self.goals_tab, text='Goals')
        self.notebook.add(self.tools_tab, text='Tool History')
        self.notebook.add(self.personality_tab, text='Personality')
        self.notebook.add(self.journal_tab, text='Journal')
        
        # Initialize tab contents
        self.setup_memory_tab()
        self.setup_emotion_tab()
        self.setup_emotion_graph_tab()
        self.setup_goals_tab()
        self.setup_tools_tab()
        self.setup_personality_tab()
        self.setup_journal_tab()
        
        # Create reload button
        self.reload_btn = ttk.Button(self.main_container, text="Reload Data", command=self.reload_all_data)
        self.reload_btn.pack(pady=5)
        
        # Start auto-reload thread
        self.auto_reload = True
        self.reload_thread = threading.Thread(target=self.auto_reload_data, daemon=True)
        self.reload_thread.start()
        
    def setup_memory_tab(self):
        # Create main frames
        control_frame = ttk.Frame(self.memory_tab)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        content_frame = ttk.Frame(self.memory_tab)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add memory type selector
        self.memory_type_var = tk.StringVar(value="long_term")
        ttk.Radiobutton(control_frame, text="Long-term", variable=self.memory_type_var, 
                       value="long_term", command=self.update_memory_display).pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Short-term", variable=self.memory_type_var, 
                       value="short_term", command=self.update_memory_display).pack(side='left', padx=5)
        
        # Add search entry and button
        self.memory_search_var = tk.StringVar()
        search_entry = ttk.Entry(control_frame, textvariable=self.memory_search_var)
        search_entry.pack(side='left', expand=True, fill='x', padx=5)
        
        search_btn = ttk.Button(control_frame, text="Search", command=self.search_memories)
        search_btn.pack(side='left', padx=5)
        
        # Create table and detail view with paned window
        paned = ttk.PanedWindow(content_frame, orient='vertical')
        paned.pack(fill='both', expand=True)
        
        # Create table frame
        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=2)
        
        # Create table
        columns = ('index', 'timestamp', 'preview', 'emotions')
        self.memory_table = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Configure columns
        self.memory_table.heading('index', text='#')
        self.memory_table.heading('timestamp', text='Timestamp')
        self.memory_table.heading('preview', text='Preview')
        self.memory_table.heading('emotions', text='Emotions')
        
        self.memory_table.column('index', width=50, anchor='center')
        self.memory_table.column('timestamp', width=150, anchor='w')
        self.memory_table.column('preview', width=400, anchor='w')
        self.memory_table.column('emotions', width=200, anchor='w')
        
        # Add scrollbar to table
        table_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.memory_table.yview)
        self.memory_table.configure(yscrollcommand=table_scroll.set)
        
        # Pack table and scrollbar
        self.memory_table.pack(side='left', fill='both', expand=True)
        table_scroll.pack(side='right', fill='y')
        
        # Create detail view
        detail_frame = ttk.LabelFrame(paned, text='Memory Details')
        paned.add(detail_frame, weight=1)
        
        self.memory_detail = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, height=10)
        self.memory_detail.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bind selection event
        self.memory_table.bind('<<TreeviewSelect>>', self.show_memory_detail)
        
    def extract_timestamp(self, memory):
        """Try to extract timestamp from memory content"""
        if isinstance(memory, str):
            # Try to find timestamp in format "YYYY-MM-DD HH:MM:SS"
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', memory)
            if match:
                return match.group(0)
        return ""
        
    def get_preview(self, memory, max_length=50):
        """Get a preview of the memory content"""
        if isinstance(memory, str):
            # Remove timestamp if present
            content = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', memory).strip()
            return (content[:max_length] + '...') if len(content) > max_length else content
        return str(memory)
        
    def format_emotions(self, emotions):
        """Format emotional context for display"""
        if not emotions:
            return ""
        formatted = []
        for emotion, value in emotions.items():
            if isinstance(value, (int, float)) and value > 0.1:
                formatted.append(f"{emotion}: {value:.2f}")
        return ", ".join(formatted)
        
    def process_memories(self, memories):
        """Process memories into display format"""
        items = []
        for i, memory in enumerate(memories, 1):
            timestamp = self.extract_timestamp(memory)
            preview = self.get_preview(memory)
            emotions = self.format_emotions(self.memory_data.get('associations', {}).get(memory, {}))
            items.append((i, timestamp, preview, emotions))
        return items
        
    def update_memory_cache(self):
        """Update the memory cache if data has changed"""
        if not self.memory_data:
            return False
            
        current_data = {
            'long_term': self.memory_data.get('long_term', []),
            'short_term': self.memory_data.get('short_term', [])
        }
        
        # Check if data has changed
        if (self.memory_cache['last_update'] is None or
            len(current_data['long_term']) != len(self.memory_cache['long_term']) or
            len(current_data['short_term']) != len(self.memory_cache['short_term'])):
            
            print("Updating memory cache...")
            start_time = time.time()
            
            # Process both memory types
            self.memory_cache['long_term'] = self.process_memories(current_data['long_term'])
            self.memory_cache['short_term'] = self.process_memories(current_data['short_term'])
            self.memory_cache['last_update'] = time.time()
            
            print(f"Cache updated in {time.time() - start_time:.2f} seconds")
            return True
            
        return False
        
    def update_memory_display(self):
        print("Updating memory display...")
        start_time = time.time()
        
        # Clear existing items
        self.memory_table.delete(*self.memory_table.get_children())
        
        if not self.memory_data:
            print("No memory data available")
            return
            
        # Update cache if needed
        self.update_memory_cache()
        
        # Get cached items for the current memory type
        memory_type = self.memory_type_var.get()
        items = self.memory_cache['long_term' if memory_type == 'long_term' else 'short_term']
        
        print(f"Displaying {len(items)} memories from cache...")
        
        # Batch insert
        insert_start = time.time()
        for item in items:
            self.memory_table.insert('', 'end', values=item)
        insert_time = time.time() - insert_start
        print(f"Inserted into table in {insert_time:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total display update time: {total_time:.2f} seconds")
        
    def show_memory_detail(self, event):
        """Show full memory details when a row is selected"""
        selection = self.memory_table.selection()
        if not selection:
            return
            
        # Get the selected item
        item = self.memory_table.item(selection[0])
        index = int(item['values'][0]) - 1
        
        # Get memory type and corresponding memory
        memory_type = self.memory_type_var.get()
        memories = self.memory_data.get('long_term' if memory_type == 'long_term' else 'short_term', [])
        
        if 0 <= index < len(memories):
            memory = memories[index]
            
            # Clear and update detail view
            self.memory_detail.delete('1.0', tk.END)
            
            # Show full memory content
            self.memory_detail.insert(tk.END, f"Memory Content:\n{memory}\n\n")
            
            # Show emotional associations if available
            emotions = self.memory_data.get('associations', {}).get(memory, {})
            if emotions:
                self.memory_detail.insert(tk.END, "Emotional Context:\n")
                for emotion, value in emotions.items():
                    if isinstance(value, (int, float)) and value > 0.1:
                        self.memory_detail.insert(tk.END, f"{emotion}: {value:.2f}\n")
                        
    def search_memories(self):
        query = self.memory_search_var.get().lower()
        if not query:
            messagebox.showwarning("Search", "Please enter a search query")
            return
            
        if not self.memory_data:
            messagebox.showwarning("Search", "No memory data available")
            return
            
        print(f"Searching for: {query}")
        start_time = time.time()
        
        # Clear existing items
        self.memory_table.delete(*self.memory_table.get_children())
        
        # Get cached items for the current memory type
        memory_type = self.memory_type_var.get()
        items = self.memory_cache['long_term' if memory_type == 'long_term' else 'short_term']
        
        # Filter items based on search query
        filtered_items = []
        for item in items:
            if query in str(item).lower():
                filtered_items.append(item)
                
        print(f"Found {len(filtered_items)} matches")
        
        # Display filtered items
        for item in filtered_items:
            self.memory_table.insert('', 'end', values=item)
            
        if not filtered_items:
            messagebox.showinfo("Search", "No matching memories found")
            
        print(f"Search completed in {time.time() - start_time:.2f} seconds")
        
    def setup_emotion_tab(self):
        # Create emotion display area
        self.emotion_display = scrolledtext.ScrolledText(self.emotion_tab, wrap=tk.WORD, height=30)
        self.emotion_display.pack(expand=True, fill='both', padx=5, pady=5)
        
    def setup_emotion_graph_tab(self):
        """Setup the emotion graph tab with matplotlib plot"""
        # Create control frame
        control_frame = ttk.Frame(self.emotion_graph_tab)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Add controls for graph customization
        ttk.Label(control_frame, text="Time Range:").pack(side='left', padx=5)
        self.graph_range_var = tk.StringVar(value="all")
        ttk.Radiobutton(control_frame, text="All", variable=self.graph_range_var, 
                       value="all", command=self.update_emotion_graph).pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Last 50", variable=self.graph_range_var, 
                       value="50", command=self.update_emotion_graph).pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Last 20", variable=self.graph_range_var, 
                       value="20", command=self.update_emotion_graph).pack(side='left', padx=5)
        
        # Add status label for hover information
        self.hover_status = ttk.Label(control_frame, text="Hover over points to see details")
        self.hover_status.pack(side='right', padx=5)
        
        # Create frame for the matplotlib figure
        self.graph_frame = ttk.Frame(self.emotion_graph_tab)
        self.graph_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create initial plot
        self.create_emotion_plot()
        
    def create_emotion_plot(self):
        """Create the matplotlib figure and canvas"""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.set_facecolor('#f0f0f0')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)
        
        # Connect the mouse events directly to the figure canvas
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Create annotation for hover tooltip
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"),
                            visible=False)
        
        # Store data for interaction
        self.points = {}  # emotion_name -> (scatter_plot, indices)
        
    def on_hover(self, event):
        """Handle hover events"""
        # Check if mouse is over axis
        if event.inaxes != self.ax:
            # Hide annotation if mouse outside axis
            self.annot.set_visible(False)
            self.hover_status.config(text="Hover over points to see details")
            self.canvas.draw_idle()
            return
            
        # Find the point closest to cursor
        closest_dist = float('inf')
        closest_point = None
        closest_emotion = None
        closest_idx = None
        
        # Check each emotion series
        for emotion_name, (scatter, indices) in self.points.items():
            # Get scatter point offsets (coordinates)
            offsets = scatter.get_offsets()
            
            # For each point, calculate distance to cursor
            for i, (x, y) in enumerate(offsets):
                dist = ((x - event.xdata)**2 + (y - event.ydata)**2)**0.5
                if dist < closest_dist and dist < 1.0:  # Use a reasonable threshold
                    closest_dist = dist
                    closest_point = (x, y)
                    closest_emotion = emotion_name
                    closest_idx = indices[i]
        
        # If found a point close to the cursor
        if closest_point is not None:
            # Get the memory content
            memories = self.memory_data.get('long_term', [])
            if 0 <= closest_idx < len(memories):
                memory_content = memories[closest_idx]
                
                # Get a short preview
                preview = memory_content[:30] + "..." if len(memory_content) > 30 else memory_content
                
                # Update annotation
                self.annot.xy = closest_point
                text = f"{closest_emotion}\nMemory #{closest_idx}:\n{preview}"
                self.annot.set_text(text)
                self.annot.set_visible(True)
                
                # Update status label
                self.hover_status.config(text=f"Viewing: {closest_emotion} - Memory #{closest_idx}")
                
                # Redraw canvas
                self.canvas.draw_idle()
        else:
            # Hide annotation if no close point
            self.annot.set_visible(False)
            self.hover_status.config(text="Hover over points to see details")
            self.canvas.draw_idle()
            
    def on_click(self, event):
        """Handle click events"""
        # Check if click is on the axis
        if event.inaxes != self.ax:
            return
            
        # Find the point closest to click
        closest_dist = float('inf')
        closest_emotion = None
        closest_idx = None
        
        # Check each emotion series
        for emotion_name, (scatter, indices) in self.points.items():
            # Get scatter point offsets (coordinates)
            offsets = scatter.get_offsets()
            
            # For each point, calculate distance to click
            for i, (x, y) in enumerate(offsets):
                dist = ((x - event.xdata)**2 + (y - event.ydata)**2)**0.5
                if dist < closest_dist and dist < 1.0:  # Use a reasonable threshold
                    closest_dist = dist
                    closest_emotion = emotion_name
                    closest_idx = indices[i]
        
        # If found a point close to the click
        if closest_idx is not None:
            try:
                # Get the memory content and emotions
                memories = self.memory_data.get('long_term', [])
                associations = self.memory_data.get('associations', {})
                
                if 0 <= closest_idx < len(memories):
                    memory_content = memories[closest_idx]
                    emotions = associations.get(memory_content, {})
                    
                    # Validate data before showing popup
                    if memory_content:
                        # Show popup with memory details
                        self.show_memory_popup(closest_idx, memory_content, emotions)
                    else:
                        print(f"Warning: Empty memory content for index {closest_idx}")
                else:
                    print(f"Warning: Memory index {closest_idx} out of range")
            except Exception as e:
                print(f"Error handling click: {e}")
        
    def show_memory_popup(self, memory_index, memory_content, emotions):
        """Show a popup window with memory details"""
        try:
            popup = tk.Toplevel(self)
            popup.title(f"Memory #{memory_index}")
            popup.geometry("600x400")
            
            # Create content frame
            content_frame = ttk.Frame(popup, padding="10")
            content_frame.pack(fill='both', expand=True)
            
            # Add memory content
            ttk.Label(content_frame, text="Memory Content:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
            memory_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, height=10)
            memory_text.pack(fill='both', expand=True, pady=(0, 10))
            memory_text.insert('1.0', str(memory_content))
            memory_text.configure(state='disabled')
            
            # Add emotional context
            ttk.Label(content_frame, text="Emotional Context:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
            emotion_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, height=5)
            emotion_text.pack(fill='both', expand=True, pady=(0, 10))
            
            # Format and display emotions
            emotion_lines = []
            for emotion, value in emotions.items():
                if isinstance(value, (int, float)) and value > 0.1:
                    emotion_lines.append(f"{emotion.capitalize()}: {value:.2f}")
            emotion_text.insert('1.0', "\n".join(emotion_lines))
            emotion_text.configure(state='disabled')
            
            # Add close button
            ttk.Button(content_frame, text="Close", command=popup.destroy).pack(pady=10)
            
            # Center the window on the screen
            popup.update_idletasks()
            width = popup.winfo_width()
            height = popup.winfo_height()
            x = (popup.winfo_screenwidth() // 2) - (width // 2)
            y = (popup.winfo_screenheight() // 2) - (height // 2)
            popup.geometry(f'+{x}+{y}')
            
            # Make the window float on top AFTER it's positioned and updated
            popup.transient(self)
            
            # Ensure the window is viewable before calling grab_set
            self.update_idletasks()  # Update Tk to make sure window is mapped
            
            # Only try to grab if the window is viewable
            try:
                popup.focus_force()  # Force focus before grabbing
                popup.grab_set()
            except Exception as e:
                print(f"Warning: Could not set grab on popup: {e}")
                
        except Exception as e:
            print(f"Error creating memory popup: {e}")
        
    def update_emotion_graph(self):
        """Update the emotion graph with current data"""
        if not self.memory_data:
            return
            
        print("Updating emotion graph...")
        
        # Clear the current plot
        self.ax.clear()
        self.points = {}
        
        # Recreate annotation
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"),
                            visible=False)
        
        # Get all memories with emotional associations
        memories = self.memory_data.get('long_term', [])
        associations = self.memory_data.get('associations', {})
        
        # Collect emotional data points
        emotion_data = {}  # emotion_name -> [(index, value), ...]
        
        for i, memory in enumerate(memories):
            if memory in associations:
                emotions = associations[memory]
                for emotion_name, value in emotions.items():
                    if isinstance(value, (int, float)):
                        if emotion_name not in emotion_data:
                            emotion_data[emotion_name] = []
                        emotion_data[emotion_name].append((i, value))
        
        # Get the selected range
        range_value = self.graph_range_var.get()
        if range_value != "all":
            range_limit = int(range_value)
            if emotion_data:
                max_index = max(max(points, key=lambda x: x[0])[0] for points in emotion_data.values())
                start_index = max(0, max_index - range_limit)
                
                # Filter data points within range
                for emotion in emotion_data:
                    emotion_data[emotion] = [(i, v) for i, v in emotion_data[emotion] if i >= start_index]
        
        print(f"Plotting {len(emotion_data)} emotion series...")
        
        # Plot each emotion line
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotion_data)))
        for (emotion_name, points), color in zip(emotion_data.items(), colors):
            if not points:
                print(f"No points to plot for {emotion_name}")
                continue
                
            indices, values = zip(*points)
            
            # Plot line
            self.ax.plot(indices, values, '-', color=color, alpha=0.3)
            
            # Plot points
            scatter = self.ax.scatter(indices, values, c=[color], label=emotion_name.capitalize(), 
                                     alpha=0.7, s=80, zorder=5)
            
            # Store for interaction
            self.points[emotion_name] = (scatter, indices)
            
            print(f"Added scatter plot for {emotion_name} with {len(indices)} points")
        
        # Customize the plot
        self.ax.set_title('Emotional States Over Time\n(Hover for preview, click for details)')
        self.ax.set_xlabel('Memory Index')
        self.ax.set_ylabel('Emotion Intensity')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent label cutoff
        self.fig.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()
        print("Graph update complete")
        
    def setup_goals_tab(self):
        # Create goals display area
        self.goals_display = scrolledtext.ScrolledText(self.goals_tab, wrap=tk.WORD, height=30)
        self.goals_display.pack(expand=True, fill='both', padx=5, pady=5)
        
    def setup_tools_tab(self):
        # Create tools display area
        self.tools_display = scrolledtext.ScrolledText(self.tools_tab, wrap=tk.WORD, height=30)
        self.tools_display.pack(expand=True, fill='both', padx=5, pady=5)
        
    def setup_personality_tab(self):
        # Create personality display area
        self.personality_display = scrolledtext.ScrolledText(self.personality_tab, wrap=tk.WORD, height=30)
        self.personality_display.pack(expand=True, fill='both', padx=5, pady=5)
        
    def setup_journal_tab(self):
        # Create journal display area
        self.journal_display = scrolledtext.ScrolledText(self.journal_tab, wrap=tk.WORD, height=30)
        self.journal_display.pack(expand=True, fill='both', padx=5, pady=5)
        
    def load_memory_data(self):
        try:
            print("Loading memory data...")
            start_time = time.time()
            with open(self.memory_path, 'rb') as f:
                self.memory_data = pickle.load(f)
            load_time = time.time() - start_time
            print(f"Memory data loaded in {load_time:.2f} seconds")
            print(f"Number of memories - Long-term: {len(self.memory_data.get('long_term', []))}, Short-term: {len(self.memory_data.get('short_term', []))}")
            return True
        except Exception as e:
            print(f"Error loading memory data: {e}")
            return False
            
    def load_tool_registry_data(self):
        try:
            with open(self.tool_registry_path, 'rb') as f:
                self.tool_registry_data = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading tool registry data: {e}")
            return False
            
    def load_journal_data(self):
        try:
            with open(self.journal_path, 'r') as f:
                self.journal_data = f.read()
            return True
        except Exception as e:
            print(f"Error loading journal data: {e}")
            return False
            
    def update_emotion_display(self):
        if not self.tool_registry_data:
            self.emotion_display.delete('1.0', tk.END)
            self.emotion_display.insert(tk.END, "No emotion data available")
            return
            
        self.emotion_display.delete('1.0', tk.END)
        
        # Get emotions from the emotion center in the tool registry
        emotion_center = self.tool_registry_data.get('emotion_center', {})
        emotions = emotion_center.get('emotions', {})
        
        self.emotion_display.insert(tk.END, "Current Emotional State:\n\n")
        
        if not emotions:
            self.emotion_display.insert(tk.END, "No emotional data available\n")
            return
            
        # Display each emotion and its intensity
        for emotion_name, emotion_data in emotions.items():
            intensity = emotion_data.get('intensity', 0.0)
            self.emotion_display.insert(tk.END, f"{emotion_name.capitalize()}: {intensity:.2f}\n")
            
        # Display overall mood if available
        mood = emotion_center.get('mood', None)
        if mood is not None:
            self.emotion_display.insert(tk.END, f"\nOverall Mood: {mood:.2f} (-1 to 1 scale)\n")
            
    def update_goals_display(self):
        if not self.tool_registry_data:
            self.goals_display.delete('1.0', tk.END)
            self.goals_display.insert(tk.END, "No goals data available")
            return
            
        self.goals_display.delete('1.0', tk.END)
        goals = self.tool_registry_data.get('goal_manager', {})
        
        # Display short-term goals
        self.goals_display.insert(tk.END, "Short-term Goals:\n")
        for goal in goals.get('short_term_goals', []):
            self.goals_display.insert(tk.END, f"- {goal['goal']}\n")
            
        # Display long-term goal
        self.goals_display.insert(tk.END, "\nLong-term Goal:\n")
        if goals.get('long_term_goal'):
            self.goals_display.insert(tk.END, f"- {goals['long_term_goal']['goal']}\n")
        else:
            self.goals_display.insert(tk.END, "No long-term goal set\n")
            
    def update_tools_display(self):
        if not self.tool_registry_data:
            self.tools_display.delete('1.0', tk.END)
            self.tools_display.insert(tk.END, "No tool history available")
            return
            
        self.tools_display.delete('1.0', tk.END)
        tool_history = self.tool_registry_data.get('tool_history', [])
        
        self.tools_display.insert(tk.END, "Recent Tool Usage:\n\n")
        for entry in tool_history:
            timestamp = entry.get('timestamp', 'Unknown time')
            name = entry.get('name', 'Unknown tool')
            params = entry.get('params', {})
            result = entry.get('result', {})
            
            self.tools_display.insert(tk.END, f"Time: {timestamp}\n")
            self.tools_display.insert(tk.END, f"Tool: {name}\n")
            self.tools_display.insert(tk.END, f"Parameters: {params}\n")
            self.tools_display.insert(tk.END, f"Result: {result}\n")
            self.tools_display.insert(tk.END, "-" * 50 + "\n")
            
    def update_personality_display(self):
        if not self.tool_registry_data:
            self.personality_display.delete('1.0', tk.END)
            self.personality_display.insert(tk.END, "No personality data available")
            return
            
        self.personality_display.delete('1.0', tk.END)
        personality = self.tool_registry_data.get('personality_manager', {})
        traits = personality.get('personality_traits', [])
        
        self.personality_display.insert(tk.END, "Personality Traits:\n\n")
        for trait in traits:
            self.personality_display.insert(tk.END, f"Trait: {trait['trait']}\n")
            self.personality_display.insert(tk.END, f"Importance: {trait['importance']}\n")
            self.personality_display.insert(tk.END, f"Reinforcement Count: {trait['reinforcement_count']}\n")
            self.personality_display.insert(tk.END, "-" * 50 + "\n")
            
    def update_journal_display(self):
        if not self.journal_data:
            self.journal_display.delete('1.0', tk.END)
            self.journal_display.insert(tk.END, "No journal entries available")
            return
            
        self.journal_display.delete('1.0', tk.END)
        self.journal_display.insert(tk.END, self.journal_data)
        
    def reload_all_data(self):
        self.load_memory_data()
        self.load_tool_registry_data()
        self.load_journal_data()
        
        self.update_memory_display()
        self.update_emotion_display()
        self.update_emotion_graph()
        self.update_goals_display()
        self.update_tools_display()
        self.update_personality_display()
        self.update_journal_display()
        
    def auto_reload_data(self):
        while self.auto_reload:
            print("\nAuto-reloading data...")
            start_time = time.time()
            
            # Check if files have been modified
            try:
                memory_mtime = os.path.getmtime(self.memory_path)
                tool_mtime = os.path.getmtime(self.tool_registry_path)
                journal_mtime = os.path.getmtime(self.journal_path)
                
                # Only reload if files have been modified
                if (not self.memory_cache['last_update'] or 
                    memory_mtime > self.memory_cache['last_update']):
                    self.reload_all_data()
                    print(f"Auto-reload completed in {time.time() - start_time:.2f} seconds")
                else:
                    print("No changes detected, skipping reload")
            except Exception as e:
                print(f"Error checking file modifications: {e}")
                
            time.sleep(5)  # Reload every 5 seconds
            
    def on_closing(self):
        self.auto_reload = False
        self.destroy()

if __name__ == "__main__":
    app = AgentViewer()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
