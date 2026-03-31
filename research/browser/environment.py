"""Synthetic browser environment for developing the browser agent.

Provides a deterministic, fast simulation of web pages with interactive
elements.  No real browser is involved -- this is purely tensor-based so
the training loop can iterate quickly on CPU.

Pages:
    Each page is a grid of elements (buttons, text inputs, links, labels).
    Elements have types, positions, text content, and states (visible,
    focused, filled, clicked).

Tasks:
    - click:    click a specific element
    - type:     type text into an input field
    - navigate: click a link to go to another page
    - form:     fill a form (multiple type + click steps)

Actions:
    0=click(x,y), 1=scroll(direction), 2=type(text), 3=navigate(url),
    4=wait, 5=done

Observations:
    - screen: (C, H, W) tensor encoding element types and positions
    - dom:    (max_elements, dom_feature_dim) structured element features
    - layout: (max_elements, 4) bounding boxes [x0, y0, x1, y1]

Rewards:
    Binary task completion + partial credit for making progress.

Usage:
    env = BrowserEnvironment(seed=42)
    obs = env.reset(task_type="click")
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---- Element types ----

ELEMENT_TYPES = ["button", "input", "link", "label", "checkbox", "dropdown"]
ELEMENT_TYPE_TO_ID = {e: i + 1 for i, e in enumerate(ELEMENT_TYPES)}  # 0 = empty

ACTION_NAMES = ["click", "scroll", "type", "navigate", "wait", "done"]
NUM_ACTIONS = len(ACTION_NAMES)

TASK_TYPES = ["click", "type", "navigate", "form"]


# ---- Page element ----

@dataclass
class Element:
    """A single interactive page element."""
    etype: str                  # element type
    eid: int                    # unique id on page
    text: str                   # display / placeholder text
    x: float                    # center x in [0, 1]
    y: float                    # center y in [0, 1]
    w: float = 0.15            # width
    h: float = 0.08            # height
    visible: bool = True
    focused: bool = False
    value: str = ""             # current value (for inputs)
    clicked: bool = False
    target_page: Optional[str] = None  # for links: which page it leads to


# ---- Page ----

@dataclass
class Page:
    """A simulated web page."""
    name: str
    elements: List[Element] = field(default_factory=list)
    background_color: int = 0  # encoded as int

    def element_at(self, x: float, y: float, tolerance: float = 0.1) -> Optional[Element]:
        """Find the element closest to (x, y), within tolerance."""
        best = None
        best_dist = float("inf")
        for elem in self.elements:
            if not elem.visible:
                continue
            dx = abs(elem.x - x)
            dy = abs(elem.y - y)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < best_dist and dist < tolerance:
                best = elem
                best_dist = dist
        return best

    def get_element_by_text(self, text: str) -> Optional[Element]:
        """Find element by its text content."""
        for elem in self.elements:
            if elem.text.lower() == text.lower() and elem.visible:
                return elem
        return None


# ---- Page generator ----

class PageGenerator:
    """Generates deterministic synthetic pages."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._button_texts = [
            "Submit", "Cancel", "OK", "Next", "Back", "Login",
            "Sign Up", "Save", "Delete", "Confirm",
        ]
        self._input_labels = [
            "Email", "Password", "Name", "Phone", "Address",
            "Username", "Search", "Comment", "Title", "Message",
        ]
        self._link_texts = [
            "Home", "About", "Contact", "Settings", "Profile",
            "Help", "Dashboard", "Products", "Blog", "FAQ",
        ]

    def generate_page(self, name: str, n_elements: int = 6) -> Page:
        """Generate a page with random elements."""
        elements = []
        positions_used = []

        for i in range(n_elements):
            etype = self._rng.choice(ELEMENT_TYPES[:4])  # button, input, link, label
            x = self._rng.uniform(0.15, 0.85)
            y = self._rng.uniform(0.1, 0.9)

            # Avoid overlapping
            for _ in range(10):
                collision = False
                for px, py in positions_used:
                    if abs(x - px) < 0.18 and abs(y - py) < 0.12:
                        collision = True
                        break
                if not collision:
                    break
                x = self._rng.uniform(0.15, 0.85)
                y = self._rng.uniform(0.1, 0.9)
            positions_used.append((x, y))

            if etype == "button":
                text = self._rng.choice(self._button_texts)
            elif etype == "input":
                text = self._rng.choice(self._input_labels)
            elif etype == "link":
                text = self._rng.choice(self._link_texts)
            else:
                text = "Label {}".format(i)

            target = None
            if etype == "link":
                target = "page_{}".format(self._rng.randint(0, 4))

            elements.append(Element(
                etype=etype, eid=i, text=text,
                x=round(x, 3), y=round(y, 3),
                target_page=target,
            ))

        return Page(name=name, elements=elements)

    def generate_site(self, n_pages: int = 5) -> Dict[str, Page]:
        """Generate a small website with interlinked pages."""
        pages = {}
        for i in range(n_pages):
            name = "page_{}".format(i)
            page = self.generate_page(name, n_elements=self._rng.randint(4, 8))
            pages[name] = page
        return pages


# ---- Task specification ----

@dataclass
class Task:
    """A browser task to complete."""
    task_type: str              # click, type, navigate, form
    instruction: str            # natural language instruction
    target_element_text: str    # which element to interact with
    target_value: str = ""      # for type tasks: what to type
    target_page: str = ""       # for navigate tasks: which page
    max_steps: int = 10


class TaskGenerator:
    """Generates deterministic tasks for a given page."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._type_values = [
            "test@example.com", "password123", "John Doe",
            "555-0100", "123 Main St", "user42",
            "hello world", "great product", "My Title", "Hi there",
        ]

    def generate_task(self, page: Page, task_type: Optional[str] = None) -> Optional[Task]:
        """Generate a task for the given page.

        Returns None if no suitable task can be generated for this page.
        """
        if task_type is None:
            task_type = self._rng.choice(TASK_TYPES)

        if task_type == "click":
            buttons = [e for e in page.elements if e.etype == "button" and e.visible]
            if not buttons:
                return None
            target = self._rng.choice(buttons)
            return Task(
                task_type="click",
                instruction="Click the '{}' button".format(target.text),
                target_element_text=target.text,
                max_steps=5,
            )

        elif task_type == "type":
            inputs = [e for e in page.elements if e.etype == "input" and e.visible]
            if not inputs:
                return None
            target = self._rng.choice(inputs)
            value = self._rng.choice(self._type_values)
            return Task(
                task_type="type",
                instruction="Type '{}' into the {} field".format(value, target.text),
                target_element_text=target.text,
                target_value=value,
                max_steps=6,
            )

        elif task_type == "navigate":
            links = [e for e in page.elements
                     if e.etype == "link" and e.visible and e.target_page]
            if not links:
                return None
            target = self._rng.choice(links)
            return Task(
                task_type="navigate",
                instruction="Navigate to the '{}' page".format(target.text),
                target_element_text=target.text,
                target_page=target.target_page,
                max_steps=5,
            )

        elif task_type == "form":
            inputs = [e for e in page.elements if e.etype == "input" and e.visible]
            buttons = [e for e in page.elements if e.etype == "button" and e.visible]
            if not inputs or not buttons:
                return None
            target_input = self._rng.choice(inputs)
            target_button = self._rng.choice(buttons)
            value = self._rng.choice(self._type_values)
            return Task(
                task_type="form",
                instruction="Fill '{}' with '{}' and click '{}'".format(
                    target_input.text, value, target_button.text,
                ),
                target_element_text=target_input.text,
                target_value=value,
                max_steps=10,
            )

        return None


# ---- Observation encoding ----

SCREEN_H = 14  # matches roughly (8*8=64 -> 14x14 = 196 is close enough spatially)
SCREEN_W = 14
SCREEN_C = 4   # channels: element_type, element_id, state_flags, text_hash
DOM_MAX_ELEMENTS = 16
DOM_FEATURE_DIM = 12  # etype_id, x, y, w, h, visible, focused, clicked, value_len, ...


def encode_screen(page: Page) -> torch.Tensor:
    """Encode a page as a (C, H, W) screen tensor.

    Channel 0: element type id at each spatial position
    Channel 1: element unique id
    Channel 2: state flags (focused, clicked, filled)
    Channel 3: text hash (simple character sum mod 1)
    """
    screen = torch.zeros(SCREEN_C, SCREEN_H, SCREEN_W)

    for elem in page.elements:
        if not elem.visible:
            continue
        # Map (x, y) to grid position
        gx = int(elem.x * (SCREEN_W - 1))
        gy = int(elem.y * (SCREEN_H - 1))
        gx = max(0, min(SCREEN_W - 1, gx))
        gy = max(0, min(SCREEN_H - 1, gy))

        # Also fill a small region around the element
        half_w = max(1, int(elem.w * SCREEN_W / 2))
        half_h = max(1, int(elem.h * SCREEN_H / 2))

        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                py = gy + dy
                px = gx + dx
                if 0 <= py < SCREEN_H and 0 <= px < SCREEN_W:
                    screen[0, py, px] = ELEMENT_TYPE_TO_ID.get(elem.etype, 0)
                    screen[1, py, px] = (elem.eid + 1) / 10.0
                    state_val = 0.0
                    if elem.focused:
                        state_val += 0.3
                    if elem.clicked:
                        state_val += 0.3
                    if elem.value:
                        state_val += 0.4
                    screen[2, py, px] = state_val
                    text_hash = sum(ord(c) for c in elem.text) % 256 / 256.0
                    screen[3, py, px] = text_hash

    return screen


def encode_dom(page: Page) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode DOM elements as structured features.

    Returns:
        elements: (max_elements, dom_feature_dim) element features
        layout:   (max_elements, 4) bounding boxes [x0, y0, x1, y1]
    """
    elements = torch.zeros(DOM_MAX_ELEMENTS, DOM_FEATURE_DIM)
    layout_out = torch.zeros(DOM_MAX_ELEMENTS, 4)

    for i, elem in enumerate(page.elements[:DOM_MAX_ELEMENTS]):
        etype_id = ELEMENT_TYPE_TO_ID.get(elem.etype, 0) / len(ELEMENT_TYPES)
        text_hash = sum(ord(c) for c in elem.text) % 256 / 256.0
        value_len = len(elem.value) / 50.0  # normalize

        elements[i] = torch.tensor([
            etype_id,
            elem.x, elem.y, elem.w, elem.h,
            float(elem.visible),
            float(elem.focused),
            float(elem.clicked),
            value_len,
            text_hash,
            float(elem.target_page is not None),
            (elem.eid + 1) / 20.0,
        ])

        x0 = elem.x - elem.w / 2
        y0 = elem.y - elem.h / 2
        x1 = elem.x + elem.w / 2
        y1 = elem.y + elem.h / 2
        layout_out[i] = torch.tensor([x0, y0, x1, y1])

    return elements, layout_out


def encode_instruction(instruction: str, dim: int = 32) -> torch.Tensor:
    """Encode a natural language instruction as a fixed-size vector.

    Uses a simple bag-of-characters hash (deterministic, no model needed).
    """
    vec = torch.zeros(dim)
    for i, ch in enumerate(instruction):
        idx = (ord(ch) * (i + 1)) % dim
        vec[idx] += 1.0
    # Normalize
    norm = vec.norm()
    if norm > 0:
        vec = vec / norm
    return vec


# ---- Action representation ----

@dataclass
class BrowserAction:
    """A browser action to execute."""
    action_type: int    # index into ACTION_NAMES
    x: float = 0.5     # normalized x coordinate [0, 1]
    y: float = 0.5     # normalized y coordinate [0, 1]
    text: str = ""      # text to type

    @property
    def name(self) -> str:
        if 0 <= self.action_type < len(ACTION_NAMES):
            return ACTION_NAMES[self.action_type]
        return "unknown"


def decode_action(
    action_logits: torch.Tensor,
    coord_pred: torch.Tensor,
    text_pred: torch.Tensor,
) -> BrowserAction:
    """Decode network outputs into a BrowserAction.

    Args:
        action_logits: (num_actions,) logits for action type
        coord_pred: (2,) predicted x, y coordinates
        text_pred: (text_dim,) text embedding (decoded to char indices)
    """
    action_type = action_logits.argmax().item()
    x = torch.sigmoid(coord_pred[0]).item()
    y = torch.sigmoid(coord_pred[1]).item()

    # Simple text decoding: map embedding peaks to characters
    text_chars = []
    if action_type == 2:  # type action
        # Use top activations as character indices
        top_vals, top_idx = text_pred.abs().topk(min(20, text_pred.shape[0]))
        for idx in top_idx:
            ch_code = (idx.item() * 7 + 97) % 128
            if 32 <= ch_code < 127:
                text_chars.append(chr(ch_code))
        if not text_chars:
            text_chars = ["a"]

    return BrowserAction(
        action_type=action_type,
        x=x, y=y,
        text="".join(text_chars[:10]),
    )


# ---- Environment ----

@dataclass
class StepResult:
    """Result of an environment step."""
    screen: torch.Tensor        # (C, H, W)
    dom_elements: torch.Tensor  # (max_elements, dom_feature_dim)
    dom_layout: torch.Tensor    # (max_elements, 4)
    reward: float
    done: bool
    info: Dict[str, Any]


class BrowserEnvironment:
    """Synthetic browser environment.

    Simulates a small website with interactive pages. The agent receives
    observations (screen pixels, DOM state) and can perform actions
    (click, type, scroll, navigate, wait, done).

    Deterministic and fast -- suitable for rapid iteration on CPU.
    """

    def __init__(self, seed: int = 42, n_pages: int = 5):
        self._seed = seed
        self._rng = random.Random(seed)
        self._page_gen = PageGenerator(self._rng)
        self._task_gen = TaskGenerator(self._rng)

        # Generate the site
        self._site = self._page_gen.generate_site(n_pages)
        self._page_names = sorted(self._site.keys())

        # State
        self._current_page_name = self._page_names[0]
        self._current_task: Optional[Task] = None
        self._step_count = 0
        self._max_steps = 10
        self._done = True
        self._progress = 0.0  # partial credit tracker

    @property
    def current_page(self) -> Page:
        return self._site[self._current_page_name]

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    def reset(
        self,
        task_type: Optional[str] = None,
        page_name: Optional[str] = None,
    ) -> Tuple[StepResult, Task]:
        """Reset the environment with a new task.

        Args:
            task_type: Type of task to generate. None = random.
            page_name: Which page to start on. None = random.

        Returns:
            (initial_observation, task) tuple.
        """
        # Pick a page
        if page_name and page_name in self._site:
            self._current_page_name = page_name
        else:
            self._current_page_name = self._rng.choice(self._page_names)

        # Reset page state
        page = self.current_page
        for elem in page.elements:
            elem.focused = False
            elem.clicked = False
            elem.value = ""

        # Generate a task -- retry if this page can't support the task type
        task = None
        attempts = 0
        while task is None and attempts < 20:
            task = self._task_gen.generate_task(page, task_type)
            if task is None:
                # Try different task type
                task_type = None
                attempts += 1

        if task is None:
            # Fallback: always-completable dummy click task
            task = Task(
                task_type="click",
                instruction="Click any button",
                target_element_text="",
                max_steps=5,
            )

        self._current_task = task
        self._step_count = 0
        self._max_steps = task.max_steps
        self._done = False
        self._progress = 0.0

        obs = self._get_observation(reward=0.0, done=False)
        return obs, task

    def step(self, action: BrowserAction) -> StepResult:
        """Execute an action and return the new observation.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self._done:
            return self._get_observation(reward=0.0, done=True)

        self._step_count += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action": action.name}

        page = self.current_page
        task = self._current_task
        assert task is not None

        if action.action_type == 0:  # click
            elem = page.element_at(action.x, action.y)
            if elem is not None:
                elem.clicked = True
                info["clicked_element"] = elem.text
                info["clicked_type"] = elem.etype

                # Check task completion for click tasks
                if task.task_type == "click":
                    if elem.text.lower() == task.target_element_text.lower():
                        reward = 1.0
                        done = True
                    elif elem.etype == "button":
                        reward = 0.1  # partial credit for clicking a button

                # For form tasks: clicking submit after filling
                elif task.task_type == "form":
                    target_input = page.get_element_by_text(task.target_element_text)
                    if (target_input and target_input.value
                            and elem.etype == "button"):
                        reward = 0.5
                        if target_input.value == task.target_value:
                            reward = 1.0
                            done = True

                # For navigate tasks: clicking the right link
                elif task.task_type == "navigate":
                    if (elem.etype == "link"
                            and elem.text.lower() == task.target_element_text.lower()):
                        if elem.target_page:
                            self._current_page_name = elem.target_page
                            reward = 1.0
                            done = True
                        else:
                            reward = 0.3
                    elif elem.etype == "link":
                        reward = 0.1  # clicked a link, wrong one

                # Unfocus all, focus clicked
                for e in page.elements:
                    e.focused = False
                if elem.etype == "input":
                    elem.focused = True

        elif action.action_type == 1:  # scroll
            info["scrolled"] = True
            # Scrolling doesn't change much in our simple env
            reward = -0.01  # slight penalty for wasted action

        elif action.action_type == 2:  # type
            # Find focused element
            focused = None
            for elem in page.elements:
                if elem.focused and elem.etype == "input":
                    focused = elem
                    break

            if focused is not None:
                focused.value = action.text
                info["typed_into"] = focused.text
                info["typed_text"] = action.text

                if task.task_type == "type":
                    if focused.text.lower() == task.target_element_text.lower():
                        if action.text == task.target_value:
                            reward = 1.0
                            done = True
                        else:
                            # Partial credit for typing in right field
                            reward = 0.3
                    else:
                        reward = 0.05  # typed in wrong field

                elif task.task_type == "form":
                    if focused.text.lower() == task.target_element_text.lower():
                        if action.text == task.target_value:
                            reward = 0.5
                            self._progress = 0.5
                        else:
                            reward = 0.2
            else:
                reward = -0.05  # tried to type with nothing focused

        elif action.action_type == 3:  # navigate
            # Navigate to a specific page by index
            page_idx = int(action.x * len(self._page_names))
            page_idx = max(0, min(len(self._page_names) - 1, page_idx))
            new_page = self._page_names[page_idx]
            self._current_page_name = new_page
            info["navigated_to"] = new_page

            if task.task_type == "navigate" and new_page == task.target_page:
                reward = 1.0
                done = True
            else:
                reward = -0.05

        elif action.action_type == 4:  # wait
            reward = -0.02  # slight penalty for waiting

        elif action.action_type == 5:  # done
            done = True
            # Penalty if task not actually completed
            if reward < 0.5:
                reward = -0.5

        # Check max steps
        if self._step_count >= self._max_steps and not done:
            done = True
            reward = max(reward, -0.3)  # timeout penalty

        self._done = done
        self._progress = max(self._progress, reward)

        info["step"] = self._step_count
        info["progress"] = self._progress

        return self._get_observation(reward=reward, done=done, info=info)

    def _get_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Build the current observation."""
        page = self.current_page
        screen = encode_screen(page)
        dom_elements, dom_layout = encode_dom(page)

        return StepResult(
            screen=screen,
            dom_elements=dom_elements,
            dom_layout=dom_layout,
            reward=reward,
            done=done,
            info=info or {},
        )

    def get_expert_action(self) -> Optional[BrowserAction]:
        """Return the expert (oracle) action for the current state.

        Used for generating demonstrations (behavioral cloning).
        Returns None if no expert action is available.
        """
        task = self._current_task
        if task is None or self._done:
            return None

        page = self.current_page

        if task.task_type == "click":
            target = page.get_element_by_text(task.target_element_text)
            if target:
                return BrowserAction(action_type=0, x=target.x, y=target.y)

        elif task.task_type == "type":
            target = page.get_element_by_text(task.target_element_text)
            if target:
                if not target.focused:
                    # First click to focus
                    return BrowserAction(action_type=0, x=target.x, y=target.y)
                else:
                    # Then type
                    return BrowserAction(
                        action_type=2, x=target.x, y=target.y,
                        text=task.target_value,
                    )

        elif task.task_type == "navigate":
            target = page.get_element_by_text(task.target_element_text)
            if target:
                return BrowserAction(action_type=0, x=target.x, y=target.y)

        elif task.task_type == "form":
            target_input = page.get_element_by_text(task.target_element_text)
            if target_input:
                if not target_input.focused:
                    return BrowserAction(
                        action_type=0, x=target_input.x, y=target_input.y,
                    )
                elif not target_input.value:
                    return BrowserAction(
                        action_type=2, x=target_input.x, y=target_input.y,
                        text=task.target_value,
                    )
                else:
                    # Find submit button
                    buttons = [e for e in page.elements
                               if e.etype == "button" and e.visible]
                    if buttons:
                        btn = buttons[0]
                        return BrowserAction(
                            action_type=0, x=btn.x, y=btn.y,
                        )

        # Fallback
        return BrowserAction(action_type=5)  # done

    def clone_state(self) -> Dict:
        """Save environment state for next-page prediction."""
        page = self.current_page
        return {
            "page_name": self._current_page_name,
            "step_count": self._step_count,
            "elements": [
                {
                    "eid": e.eid, "focused": e.focused,
                    "clicked": e.clicked, "value": e.value,
                }
                for e in page.elements
            ],
        }


# ---- Demonstration generation ----

@dataclass
class Demonstration:
    """A complete expert demonstration of a task."""
    task_type: str
    instruction: str
    observations: List[Dict[str, torch.Tensor]]
    actions: List[BrowserAction]
    rewards: List[float]
    total_reward: float
    success: bool


def generate_demonstrations(
    n_demos: int = 100,
    seed: int = 42,
    task_type: Optional[str] = None,
) -> List[Demonstration]:
    """Generate expert demonstrations using the oracle policy.

    Args:
        n_demos: Number of demonstrations to generate.
        seed: Random seed for reproducibility.
        task_type: If specified, only generate this type of task.

    Returns:
        List of Demonstration objects.
    """
    demos = []
    env = BrowserEnvironment(seed=seed)

    for i in range(n_demos):
        # Cycle through task types if not specified
        if task_type is None:
            tt = TASK_TYPES[i % len(TASK_TYPES)]
        else:
            tt = task_type

        obs, task = env.reset(task_type=tt)

        observations = [{
            "screen": obs.screen.clone(),
            "dom_elements": obs.dom_elements.clone(),
            "dom_layout": obs.dom_layout.clone(),
        }]
        actions = []
        rewards = []
        total_reward = 0.0

        for step in range(task.max_steps):
            action = env.get_expert_action()
            if action is None:
                action = BrowserAction(action_type=5)  # done

            result = env.step(action)
            actions.append(action)
            rewards.append(result.reward)
            total_reward += result.reward

            observations.append({
                "screen": result.screen.clone(),
                "dom_elements": result.dom_elements.clone(),
                "dom_layout": result.dom_layout.clone(),
            })

            if result.done:
                break

        demos.append(Demonstration(
            task_type=task.task_type,
            instruction=task.instruction,
            observations=observations,
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            success=total_reward >= 0.9,
        ))

    return demos


def batch_demonstrations(
    demos: List[Demonstration],
    d_model: int = 128,
    instruction_dim: int = 32,
) -> Dict[str, torch.Tensor]:
    """Convert demonstrations to batched tensors for training.

    Returns dict with keys:
        screens:       (N, C, H, W)
        next_screens:  (N, C, H, W)
        dom_elements:  (N, max_elem, dom_feat)
        dom_layouts:   (N, max_elem, 4)
        instructions:  (N, instruction_dim)
        action_types:  (N,) LongTensor
        action_coords: (N, 2)
        action_texts:  (N, instruction_dim)  -- encoded text
        rewards:       (N,)
        dones:         (N,) -- whether this step completed the task

    where N is the total number of (state, action, next_state) transitions.
    """
    screens = []
    next_screens = []
    dom_elems = []
    dom_lays = []
    instructions = []
    action_types = []
    action_coords = []
    action_texts = []
    rewards_list = []
    dones = []

    for demo in demos:
        inst_vec = encode_instruction(demo.instruction, dim=instruction_dim)

        for t in range(len(demo.actions)):
            obs = demo.observations[t]
            next_obs = demo.observations[t + 1]
            action = demo.actions[t]

            screens.append(obs["screen"])
            next_screens.append(next_obs["screen"])
            dom_elems.append(obs["dom_elements"])
            dom_lays.append(obs["dom_layout"])
            instructions.append(inst_vec)

            action_types.append(action.action_type)
            action_coords.append(torch.tensor([action.x, action.y]))
            action_texts.append(
                encode_instruction(action.text, dim=instruction_dim)
            )
            rewards_list.append(demo.rewards[t])
            dones.append(float(t == len(demo.actions) - 1 and demo.success))

    return {
        "screens": torch.stack(screens),
        "next_screens": torch.stack(next_screens),
        "dom_elements": torch.stack(dom_elems),
        "dom_layouts": torch.stack(dom_lays),
        "instructions": torch.stack(instructions),
        "action_types": torch.tensor(action_types, dtype=torch.long),
        "action_coords": torch.stack(action_coords),
        "action_texts": torch.stack(action_texts),
        "rewards": torch.tensor(rewards_list),
        "dones": torch.tensor(dones),
    }
