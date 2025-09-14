from dataclasses import dataclass
from typing import List, Optional
import os
import yaml

@dataclass
class TopicSpec:
    name: str
    type: Optional[str] = None

@dataclass
class PolicySpec:
    obs_topics: List[TopicSpec]
    action_topic: TopicSpec
    rate_hz: float = 20.0

@dataclass
class Contract:
    name: str
    version: int
    storage: str = 'mcap'
    bag_base_dir: str = '/tmp/episodes'
    seconds: int = 10
    record_topics: List[str] = None
    policy: Optional[PolicySpec] = None

def load_contract(path: str) -> Contract:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    record_topics = data.get('record_topics', []) or []

    policy = None
    if 'policy' in data and data['policy']:
        p = data['policy']
        obs = [TopicSpec(**t) for t in p.get('obs_topics', [])]
        action = TopicSpec(**p['action_topic']) if 'action_topic' in p else None
        rate = float(p.get('rate_hz', 20.0))
        policy = PolicySpec(obs_topics=obs, action_topic=action, rate_hz=rate)

    return Contract(
        name=data.get('name', 'contract'),
        version=int(data.get('version', 1)),
        storage=data.get('storage', 'mcap'),
        bag_base_dir=data.get('bag_base_dir', '/tmp/episodes'),
        seconds=int(data.get('seconds', 10)),
        record_topics=record_topics,
        policy=policy,
    )

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
